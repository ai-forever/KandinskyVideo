import torch
from torch import nn, einsum
from einops import rearrange

from .nn import Identity, Attention, SinusoidalPosEmb, ConditionalGroupNorm, TemporalAttention
from .utils import exist, set_default_item, set_default_layer
import torch.nn.functional as F

class TemporalConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.temporal_merge = torch.nn.Parameter(torch.tensor([0.]))
        self.temporal_projection = nn.Conv3d(in_channels, out_channels, (3, 1, 1), padding=(1, 0, 0))
        
        self.temporal_projection.weight.data.zero_()
        self.temporal_projection.bias.data.zero_()
        
    def forward(self, x, bs, temporal_mask):
        height, width = x.shape[-2:]
        temporal_mask_input = temporal_mask
        while temporal_mask_input.shape[0]>x.shape[0]:
            temporal_mask_input = temporal_mask_input[::2]

        out = rearrange(x, '(b t) c h w -> b c t h w', b=bs)
        out = self.temporal_projection(out)
        out = rearrange(out, 'b c t h w -> (b t) c h w')                
        temporal_merge = self.temporal_merge * temporal_mask_input[:, None, None, None]
        x = (1-temporal_merge)*x + temporal_merge*out
        return x

class CompressTemporal(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=(2, 1, 1),stride=(2,1,1),padding=(0,0,0))

    def forward(self, x, bs):
        hheight, width = x.shape[-2:]
        x = rearrange(x, '(b t) c h w -> b c t h w', b=bs)
        x = self.downsample(x)
        x = rearrange(x, 'b c t h w -> (b t) c h w')                
        return x

class DecompressTemporal(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.upsample = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(2, 1, 1),stride=(2,1,1),padding=(0,0,0))
        
    def forward(self, x, bs):
        height, width = x.shape[-2:]
        x = rearrange(x, '(b t) c h w -> b c t h w', b=bs)
        x = self.upsample(x)
        x = rearrange(x, 'b c t h w -> (b t) c h w')                
        return x

class Block(nn.Module):

    def __init__(
            self, in_channels, out_channels, time_embed_dim, kernel_size=3, norm_groups=32,
            up_resolution=None, temporal=False, temporal_type='conv1D', time_downsample=False, new_params=False
    ):
        super().__init__()
        self.time_downsample = time_downsample
        self.group_norm = ConditionalGroupNorm(norm_groups, in_channels, time_embed_dim)
        self.activation = nn.SiLU()
        self.up_sample = set_default_layer(
            exist(up_resolution) and up_resolution,
            nn.ConvTranspose2d, (in_channels, in_channels), {'kernel_size': 2, 'stride': 2}
        )
                        
        padding = set_default_item(kernel_size == 1, 0, 1)
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.down_sample = set_default_layer(
            exist(up_resolution) and not up_resolution,
            nn.Conv2d, (out_channels, out_channels), {'kernel_size': 2, 'stride': 2}
        )
        
        self.temporal_decompress = set_default_layer(
            exist(up_resolution) and up_resolution and time_downsample,
            DecompressTemporal, (in_channels,), {}, layer_2=Identity
        )

        self.temporal_post_upsample = set_default_layer(
            new_params and exist(up_resolution) and up_resolution and not time_downsample,
            TemporalConv3D, (out_channels,out_channels), {}, layer_2=Identity
        )

        self.temporal_projection_new = set_default_layer(
            temporal,
            TemporalConv3D, (out_channels,out_channels), {}, layer_2=Identity
        )

        self.temporal_compress = set_default_layer(
            exist(up_resolution) and not up_resolution and time_downsample,
            CompressTemporal, (out_channels,), {}, layer_2=Identity
        )
        
        self.temporal_post_downsample = set_default_layer(
            new_params and exist(up_resolution) and not up_resolution and not time_downsample,
            TemporalConv3D, (out_channels,out_channels), {}, layer_2=Identity
        )

    def forward(self, x, time_embed, bs=None, temporal_mask=None):
        x = self.group_norm(x, time_embed)
        x = self.activation(x)

        x = self.temporal_decompress(x, bs)
        x = self.up_sample(x)
        x = self.temporal_post_upsample(x, bs, temporal_mask)

        x = self.projection(x)
        x = self.temporal_projection_new(x, bs, temporal_mask)
         
        x = self.down_sample(x)
        x = self.temporal_post_downsample(x, bs, temporal_mask)
        x = self.temporal_compress(x, bs)

        return x


class ResNetBlock(nn.Module):

    def __init__(
            self, in_channels, out_channels, time_embed_dim, norm_groups=32, compression_ratio=2,
            up_resolutions=4*[None], temporal=False, temporal_type='conv1D', time_downsample=False, 
            new_params=False,
    ):
        super().__init__()
        kernel_sizes = [1, 3, 3, 1]
        hidden_channel = max(in_channels, out_channels) // compression_ratio
        hidden_channels = [(in_channels, hidden_channel)] + [(hidden_channel, hidden_channel)] * 2 + [(hidden_channel, out_channels)]
        temporals = [temporal, temporal, temporal, temporal]
        self.resnet_blocks = nn.ModuleList([
            Block(in_channel, out_channel, time_embed_dim, kernel_size, norm_groups, up_resolution, temporal, temporal_type, time_downsample=time_downsample, new_params=new_params)
            for (in_channel, out_channel), kernel_size, up_resolution, temporal in zip(
                hidden_channels, kernel_sizes, up_resolutions, temporals
            )
        ])
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shortcut_up_sample = set_default_layer(
            True in up_resolutions,
            nn.ConvTranspose2d, (in_channels, in_channels), {'kernel_size': 2, 'stride': 2}
        )
        self.shortcut_projection = set_default_layer(
            in_channels != out_channels,
            nn.Conv2d, (in_channels, out_channels), {'kernel_size': 1}
        )

        self.shortcut_down_sample = set_default_layer(
            False in up_resolutions,
            nn.Conv2d, (out_channels, out_channels), {'kernel_size': 2, 'stride': 2}
        )               
            
        self.temporal_decompress = set_default_layer(
            True in up_resolutions and time_downsample,
            DecompressTemporal, (in_channels,), {}, layer_2=Identity
        )

        self.temporal_post_upsample = set_default_layer(
            True in up_resolutions and not time_downsample and new_params,
            TemporalConv3D, (in_channels,out_channels), {}, layer_2=Identity
        )

        self.temporal_compress = set_default_layer(
            False in up_resolutions and time_downsample,
            CompressTemporal, (out_channels,), {}, layer_2=Identity
        )
        
        self.temporal_post_downsample = set_default_layer(
            False in up_resolutions and not time_downsample and new_params,
            TemporalConv3D, (out_channels,out_channels), {}, layer_2=Identity
        )


    def forward(self, x, time_embed, bs=None, temporal_mask=None):
        out = x
        for resnet_block in self.resnet_blocks:
            out = resnet_block(out, time_embed, bs, temporal_mask)

        x = self.temporal_decompress(x, bs)
        x = self.shortcut_up_sample(x)
        x = self.temporal_post_upsample(x, bs, temporal_mask)
        
        x = self.shortcut_projection(x)
        
        x = self.shortcut_down_sample(x)
        x = self.temporal_post_upsample(x, bs, temporal_mask)
        x = self.temporal_compress(x, bs)

        x = x + out
        
        return x


class AttentionPolling(nn.Module):

    def __init__(self, num_channels, context_dim, head_dim=64):
        super().__init__()
        self.attention = Attention(context_dim, num_channels, context_dim, head_dim)

    def forward(self, x, context, context_mask=None):
        context = self.attention(context.mean(dim=1, keepdim=True), context, context_mask)
        return x + context.squeeze(1)


class AttentionBlock(nn.Module):

    def __init__(
            self, num_channels, time_embed_dim, context_dim=None,
            norm_groups=32, head_dim=64, expansion_ratio=4, temporal=False
    ):
        super().__init__()
        self.in_norm = ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        self.attention = Attention(num_channels, num_channels, context_dim or num_channels, head_dim)

        hidden_channels = expansion_ratio * num_channels
        self.out_norm = ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, num_channels, kernel_size=1, bias=False),
        )

    def forward(self, x, time_embed, context=None, context_mask=None, temporal_embed=None, bs=None):
        
        while context is not None and context.shape[0]>x.shape[0]:
            context = context[::2]
        while context_mask is not None and context_mask.shape[0]>x.shape[0]:
            context_mask = context_mask[::2]

        height, width = x.shape[-2:]
        out = self.in_norm(x, time_embed)
        out = rearrange(out, 'b c h w -> b (h w) c', h=height, w=width)
        context = set_default_item(exist(context), context, out)
        out = self.attention(out, context, context_mask)
        out = rearrange(out, 'b (h w) c -> b c h w', h=height, w=width)
        x = x + out

        out = self.out_norm(x, time_embed)
        out = self.feed_forward(out)
        x = x + out
        return x

class TemporalAttentionBlock(nn.Module):
    def __init__(self, num_channels, time_embed_dim, norm_groups=32, head_dim=64, expansion_ratio=4):
        super().__init__()
        self.in_norm = ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        self.attention = Attention(num_channels, num_channels, num_channels, head_dim)

        hidden_channels = expansion_ratio * num_channels
        self.out_norm = ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, num_channels, kernel_size=1, bias=False),
        )
        self.temporal_merge1 = torch.nn.Parameter(torch.tensor([0.]))
        self.temporal_merge2 = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, x, time_embed, temporal_mask, bs):
        temporal_mask_input = temporal_mask
        while temporal_mask_input.shape[0]>x.shape[0]:
            temporal_mask_input = temporal_mask_input[::2]
        height, width = x.shape[-2:]
        
        out = self.in_norm(x, time_embed)
        out = rearrange(out, '(b t) c h w -> (b h w) t c', b=bs)
        out = self.attention(out, out)
        out = rearrange(out, '(b h w) t c -> (b t) c h w', h=height, w=width)
        
        temporal_merge = self.temporal_merge1 * temporal_mask_input[:, None, None, None]
        x = (1-temporal_merge)*x + temporal_merge*out
        
        out = self.out_norm(x, time_embed)
        out = self.feed_forward(out)
        temporal_merge = self.temporal_merge2 * temporal_mask_input[:, None, None, None]
        x = (1-temporal_merge)*x + temporal_merge*out
        return x


class DownSampleBlock(nn.Module):

    def __init__(
            self, in_channels, out_channels, time_embed_dim, context_dim=None,
            num_blocks=3, groups=32, head_dim=64, expansion_ratio=4, compression_ratio=2,
            down_sample=True, self_attention=True, temporal=False, temporal_type='conv1D', time_downsample=False, new_params=False, add_temporal_attention=False):
        super().__init__()
        self.self_attention_block = set_default_layer(
            self_attention,
            AttentionBlock,
            (in_channels, time_embed_dim, None, groups, head_dim, expansion_ratio, temporal),
            layer_2=Identity
        )

        self.temporal_attention_block = set_default_layer(
            self_attention and add_temporal_attention,
            TemporalAttentionBlock,
            (in_channels, time_embed_dim, groups, head_dim, expansion_ratio),
            layer_2=Identity
        )

        up_resolutions = [[None] * 4] * (num_blocks - 1) + [[None, None, set_default_item(down_sample, False), None]]
        time_downsample_blocks = [False] * (num_blocks-1) + [time_downsample]
        hidden_channels = [(in_channels, out_channels)] + [(out_channels, out_channels)] * (num_blocks - 1)
        self.resnet_attn_blocks = nn.ModuleList([
            nn.ModuleList([
                ResNetBlock(in_channel, out_channel, time_embed_dim, groups, compression_ratio, temporal=temporal, temporal_type=temporal_type, time_downsample=time_downsample, new_params=new_params),
                set_default_layer(
                    exist(context_dim),
                    AttentionBlock,
                    (out_channel, time_embed_dim, context_dim, groups, head_dim, expansion_ratio, temporal),
                    layer_2=Identity
                ),
                ResNetBlock(out_channel, out_channel, time_embed_dim, groups, compression_ratio, up_resolution, temporal, temporal_type=temporal_type, time_downsample=time_downsample, new_params=new_params),
            ]) for (in_channel, out_channel), up_resolution, time_downsample in zip(hidden_channels, up_resolutions, time_downsample_blocks)
        ])

    def forward(self, x, time_embed, context=None, context_mask=None, temporal_embed=None, bs=None, temporal_mask=None):
        x = self.self_attention_block(x, time_embed, temporal_embed=temporal_embed, bs=bs)
        x = self.temporal_attention_block(x, time_embed, temporal_mask, bs)
        for in_resnet_block, attention, out_resnet_block in self.resnet_attn_blocks:
            x = in_resnet_block(x, time_embed, bs, temporal_mask)
            x = attention(x, time_embed, context, context_mask, temporal_embed, bs)
            x = out_resnet_block(x, time_embed, bs, temporal_mask)
        return x


class UpSampleBlock(nn.Module):

    def __init__(
            self, in_channels, cat_dim, out_channels, time_embed_dim, context_dim=None,
            num_blocks=3, groups=32, head_dim=64, expansion_ratio=4, compression_ratio=2,
            up_sample=True, self_attention=True, temporal=False, temporal_type="conv1D", time_downsample=False, new_params=False, add_temporal_attention=False,
    ):
        super().__init__()
        up_resolutions = [[None, set_default_item(up_sample, True), None, None]] + [[None] * 4] * (num_blocks - 1)
        time_upsample_blocks = [time_downsample] + [False] * (num_blocks-1)
        hidden_channels = [(in_channels + cat_dim, in_channels)] + [(in_channels, in_channels)] * (num_blocks - 2) + [(in_channels, out_channels)]
        self.resnet_attn_blocks = nn.ModuleList([
            nn.ModuleList([
                ResNetBlock(in_channel, in_channel, time_embed_dim, groups, compression_ratio, up_resolution, temporal, temporal_type, time_downsample=time_downsample, new_params=new_params),
                set_default_layer(
                    exist(context_dim),
                    AttentionBlock,
                    (in_channel, time_embed_dim, context_dim, groups, head_dim, expansion_ratio, temporal),
                    layer_2=Identity
                ),
                ResNetBlock(in_channel, out_channel, time_embed_dim, groups, compression_ratio, temporal=temporal, temporal_type=temporal_type, new_params=new_params),
            ]) for (in_channel, out_channel), up_resolution, time_downsample in zip(hidden_channels, up_resolutions, time_upsample_blocks)
        ])

        self.self_attention_block = set_default_layer(
            self_attention,
            AttentionBlock,
            (out_channels, time_embed_dim, None, groups, head_dim, expansion_ratio, temporal),
            layer_2=Identity
        )
        self.temporal_attention_block = set_default_layer(
            self_attention and add_temporal_attention,
            TemporalAttentionBlock,
            (out_channels, time_embed_dim, groups, head_dim, expansion_ratio),
            layer_2=Identity
        )

    def forward(self, x, time_embed, context=None, context_mask=None, temporal_embed=None, bs=None, temporal_mask=None):
        for in_resnet_block, attention, out_resnet_block in self.resnet_attn_blocks:
            x = in_resnet_block(x, time_embed, bs, temporal_mask)
            x = attention(x, time_embed, context, context_mask, temporal_embed, bs)
            x = out_resnet_block(x, time_embed, bs, temporal_mask)
        x = self.self_attention_block(x, time_embed, temporal_embed=temporal_embed, bs=bs)
        x = self.temporal_attention_block(x, time_embed, temporal_mask, bs)
        return x

class UNet(nn.Module):

    def __init__(self,
                 model_channels,
                 init_channels=None,
                 num_channels=3,
                 out_channels=3,
                 time_embed_dim=None,
                 context_dim=None,
                 groups=32,
                 head_dim=64,
                 expansion_ratio=4,
                 compression_ratio=2,
                 dim_mult=(1, 2, 4, 8),
                 num_blocks=(3, 3, 3, 3),
                 add_cross_attention=(False, True, True, True),
                 add_self_attention=(False, True, True, True),
                 time_downsample=(False, False, False, False),
                 temporal=True,
                 use_perturbation_time_embed=True,
                 merge_perturbation_time_with_diffusion_time=True,
                 merge_perturbation_time_with_context=False,
                 pass_skip_frames=True,
                 temporal_type="conv3D", 
                 add_new_temporal_params=False,
                 add_temporal_attention=True,
     ):
        super().__init__()
        
        init_channels = init_channels or model_channels
        self.to_time_embed = nn.Sequential(
            SinusoidalPosEmb(init_channels),
            nn.Linear(init_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Conditioning frame perturbation (Align your latents)
        self.use_perturbation_time_embed=use_perturbation_time_embed
        self.merge_perturbation_time_with_diffusion_time = merge_perturbation_time_with_diffusion_time
        if merge_perturbation_time_with_diffusion_time and use_perturbation_time_embed:
            self.perturbation_to_time_embed = nn.Sequential(
                SinusoidalPosEmb(init_channels),
                nn.Linear(init_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim)
            )
            self.time_embed_merge = torch.nn.Parameter(torch.tensor([0.]))
        
        self.merge_perturbation_time_with_context=merge_perturbation_time_with_context
        if merge_perturbation_time_with_context and use_perturbation_time_embed: # merge with context
            self.time_embed_context_merge = torch.nn.Parameter(torch.tensor([0.]))
            self.perturbation_time_encoding = SinusoidalPosEmb(context_dim)
        
        self.pass_skip_frames = pass_skip_frames
        if self.pass_skip_frames:
            self.skip_merge = torch.nn.Parameter(torch.tensor([0.]))
            self.skip_embeddings=torch.nn.Embedding(20, time_embed_dim)

        self.feature_pooling = AttentionPolling(time_embed_dim, context_dim, head_dim)

        self.in_layer = nn.Conv2d(num_channels, init_channels, kernel_size=3, padding=1)

        hidden_dims = [init_channels, *map(lambda mult: model_channels * mult, dim_mult)]
        in_out_dims = list(zip(hidden_dims[:-1], hidden_dims[1:]))
        text_dims = [set_default_item(is_exist, context_dim) for is_exist in add_cross_attention]
        layer_params = [num_blocks, text_dims, add_self_attention, time_downsample]
        rev_layer_params = map(reversed, layer_params)

        cat_dims = []
        self.num_levels = len(in_out_dims)
        self.down_samples = nn.ModuleList([])
        for level, ((in_dim, out_dim), res_block_num, text_dim, self_attention, time_downsample_layer) in enumerate(zip(in_out_dims, *layer_params)):
            down_sample = level != (self.num_levels - 1)
            cat_dims.append(set_default_item(level != (self.num_levels - 1), out_dim, 0))
            self.down_samples.append(
                DownSampleBlock(
                    in_dim, out_dim, time_embed_dim, text_dim, res_block_num, groups, head_dim, expansion_ratio,
                    compression_ratio, down_sample, self_attention, temporal, temporal_type, time_downsample_layer, add_new_temporal_params, add_temporal_attention
                )
            )

        self.up_samples = nn.ModuleList([])
        for level, ((out_dim, in_dim), res_block_num, text_dim, self_attention, time_upsample_layer) in enumerate(zip(reversed(in_out_dims), *rev_layer_params)):
            up_sample = level != 0
            self.up_samples.append(
                UpSampleBlock(
                    in_dim, cat_dims.pop(), out_dim, time_embed_dim, text_dim, res_block_num, groups, head_dim,
                    expansion_ratio, compression_ratio, up_sample, self_attention, temporal, temporal_type, time_upsample_layer, add_new_temporal_params, add_temporal_attention
                )
            )
        
        self.out_layer = nn.Sequential(
            nn.GroupNorm(groups, init_channels),
            nn.SiLU(),
            nn.Conv2d(init_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.temporal_conv_in = set_default_layer(
            add_new_temporal_params,
            TemporalConv3D, (init_channels, init_channels), {}, layer_2=Identity
        )
        self.temporal_conv_out = set_default_layer(
            add_new_temporal_params,
            TemporalConv3D, (init_channels, init_channels), {}, layer_2=Identity
        )


    def forward(
        self, 
        x, 
        time, 
        context=None, 
        context_mask=None, 
        temporal_embed=None, 
        num_predicted_groups=None, 
        perturbation_time=None, 
        skip_frames=None,
        temporal_mask=None
    ):
        bs=None
        if num_predicted_groups is not None:
            bs=x.shape[0]//num_predicted_groups
        if perturbation_time is None:
            perturbation_time=torch.zeros(size=(context.shape[0],), device=context.device) + 25
        if skip_frames is None:
            skip_frames=torch.zeros(size=(context.shape[0],), dtype=torch.int32, device=context.device) + 1

        time_embed = self.to_time_embed(time)
        if self.use_perturbation_time_embed and self.merge_perturbation_time_with_diffusion_time:
            perturbation_time_embed = self.perturbation_to_time_embed(perturbation_time)
            time_embed = self.time_embed_merge*perturbation_time_embed+(1-self.time_embed_merge)*time_embed
            
        if self.use_perturbation_time_embed and self.merge_perturbation_time_with_context:
            perturbation_emb = self.perturbation_time_encoding(perturbation_time)
            perturbation_emb = torch.unsqueeze(perturbation_emb, dim=1)
            context = context + self.time_embed_context_merge * perturbation_emb
        
        if self.pass_skip_frames:
            skip_emb = self.skip_embeddings(skip_frames)
            time_embed = self.skip_merge*skip_emb + time_embed

        if exist(context):
            time_embed = self.feature_pooling(time_embed, context, context_mask)

        hidden_states = []
        x = self.in_layer(x)
        x = self.temporal_conv_in(x, bs, temporal_mask)
        
        for level, down_sample in enumerate(self.down_samples):
            x = down_sample(x, time_embed, context, context_mask, temporal_embed, bs, temporal_mask)
            if level != self.num_levels - 1:
                hidden_states.append(x)
        for level, up_sample in enumerate(self.up_samples):
            if level != 0:
                x = torch.cat([x, hidden_states.pop()], dim=1)
            x = up_sample(x, time_embed, context, context_mask, temporal_embed, bs, temporal_mask)
        
        x = self.temporal_conv_out(x, bs, temporal_mask)
        x = self.out_layer(x)

        return x


def get_unet(conf):
    unet = UNet(**conf)
    return unet


def get_unet(conf):
    unet = UNet(**conf)
    return unet

