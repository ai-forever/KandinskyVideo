import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

from .utils import freeze


def nonlinearity(x):
    return x*torch.sigmoid(x)


class SpatialNorm(nn.Module):
    def __init__(
        self, f_channels, zq_channels=None, norm_layer=nn.GroupNorm, freeze_norm_layer=False, add_conv=False, **norm_layer_params
    ):
        super().__init__()
        self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)
        if zq_channels is not None:
            if freeze_norm_layer:
                for p in self.norm_layer.parameters:
                    p.requires_grad = False
            self.add_conv = add_conv
            if self.add_conv:
                self.conv = nn.Conv2d(zq_channels, zq_channels, kernel_size=3, stride=1, padding=1)
            self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
            self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, inp, zq_inp=None):
        temp = []
        for i in range(inp.shape[0]):
            f = inp[i][None]
            
            if zq_inp is None:
                zq = None
            else:
                zq = zq_inp[i][None]
                
            norm_f = self.norm_layer(f)
            if zq is not None:
                f_size = f.shape[-2:]
                zq = torch.nn.functional.interpolate(zq, size=f_size, mode="nearest")
                if self.add_conv:
                    zq = self.conv(zq)
                norm_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
            temp.append(norm_f)
        norm_f = torch.cat(temp)
        return norm_f


def Normalize(in_channels, zq_ch=None, add_conv=None):
    return SpatialNorm(
            in_channels, zq_ch, norm_layer=nn.GroupNorm,
            freeze_norm_layer=False, add_conv=add_conv, num_groups=32, eps=1e-6, affine=True
        )

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, inp):
        temp = []
        for i in range(inp.shape[0]):
            x = inp[i][None]
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
            if self.with_conv:
                x = self.conv(x)
            temp.append(x)
        x = torch.cat(temp)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, inp):
        temp = []
        for i in range(inp.shape[0]):
            x = inp[i][None]
            if self.with_conv:
                pad = (0,1,0,1)
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
                x = self.conv(x)
            else:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            temp.append(x)
        x = torch.cat(temp)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, zq_ch=None, add_conv=False, temporal=False, 
                 temporal_kernel_size=(3,3,3), temporal_conv_padding=(1,1,1),
                 use_3d_conv=False, flicker_augmentation=False):
        super().__init__()
        
        self.use_3d_conv = use_3d_conv
        conv = nn.Conv3d if self.use_3d_conv else nn.Conv2d
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, zq_ch, add_conv=add_conv)
        self.conv1 = conv(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)            
        self.norm2 = Normalize(out_channels, zq_ch, add_conv=add_conv)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)
                
        self.flicker_augmentation = flicker_augmentation
        self.temporal = temporal
        if temporal:
            self.temporal_merge1 = torch.nn.Parameter(torch.tensor([0.]))
            self.temporal_projection1 = nn.Conv3d(out_channels, out_channels, temporal_kernel_size, padding=temporal_conv_padding)
            
            self.temporal_merge2 = torch.nn.Parameter(torch.tensor([0.]))
            self.temporal_projection2 = nn.Conv3d(out_channels, out_channels, temporal_kernel_size, padding=temporal_conv_padding)

            self.temporal_projection1.weight.data.zero_()
            self.temporal_projection1.bias.data.zero_()

            self.temporal_projection2.weight.data.zero_()
            self.temporal_projection2.bias.data.zero_()
            
            if flicker_augmentation: # I added it here, so that it can be trained with temporal layers
                self.temporal_flicker_proj = torch.nn.Linear(1, out_channels)

    def forward(self, x, temb, zq=None, num_frames=None, aug_emb=None):
        h_temp = []
        
        for i in range(x.shape[0]):
            h_i = x[i][None]
            if zq is not None:
                zq_i = zq[i][None]
            else:
                zq_i = None
            h_i = self.norm1(h_i, zq_i)
            h_i = nonlinearity(h_i)
            h_i = self.conv1(h_i)
            h_temp.append(h_i)
            
        h = torch.cat(h_temp)

        if num_frames is not None and self.temporal:
            out = h
            if self.flicker_augmentation and aug_emb is not None:
                # aug_emb: 1 x num_frames x 1
                aug_proj = self.temporal_flicker_proj(aug_emb)
                aug_proj = rearrange(aug_proj, 'b t c -> (b t) c')
                out = out + aug_proj[..., None, None]

            out = rearrange(out, '(b t) c h w -> b c t h w', t=num_frames)
            out = self.temporal_projection1(out)
            out = rearrange(out, 'b c t h w -> (b t) c h w')
                
            h = (1-self.temporal_merge1)*h + self.temporal_merge1*out
            
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h_temp = []
        for i in range(h.shape[0]):
            h_i = h[i][None]
            if zq is None:
                zq_i = None
            else:
                zq_i = zq[i][None]
                
            if temb is not None:
                h_i = h_i + self.temb_proj(nonlinearity(temb))[:,:,None,None]
            h_i = self.norm2(h_i, zq_i)
            h_i = nonlinearity(h_i)
            h_i = self.dropout(h_i)            
            h_i = self.conv2(h_i)
            h_temp.append(h_i)
        h = torch.cat(h_temp)

        if num_frames and self.temporal:
            out = h
            if self.flicker_augmentation and aug_emb is not None:
                # aug_emb: 1 x num_frames x 1
                aug_proj = self.temporal_flicker_proj(aug_emb)
                aug_proj = rearrange(aug_proj, 'b t c -> (b t) c')
                out = out + aug_proj[..., None, None]

            out = rearrange(out, '(b t) c h w -> b c t h w', t=num_frames)
            out = self.temporal_projection2(out)
            out = rearrange(out, 'b c t h w -> (b t) c h w')                
            
            h = (1-self.temporal_merge2)*h + self.temporal_merge2*out

        if self.in_channels != self.out_channels:
            x_temp = []
            for i in range(x.shape[0]):
                x_i = x[i][None]
                if self.use_conv_shortcut:
                    x_i = self.conv_shortcut(x_i)
                else:
                    x_i = self.nin_shortcut(x_i)
                x_temp.append(x_i)
            x=torch.cat(x_temp)
        return x+h

class AttnBlock(nn.Module):
    def __init__(self, in_channels, zq_ch=None, add_conv=False):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, zq_ch, add_conv=add_conv)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x, zq=None):
        temp = []
        for i in range(x.shape[0]):
            h_ = x[i][None]
            h_ = self.norm(h_, zq)
            q = self.q(h_)
            k = self.k(h_)
            v = self.v(h_)

            # compute attention
            b,c,h,w = q.shape
            q = q.reshape(b,c,h*w)
            q = q.permute(0,2,1)   # b,hw,c
            k = k.reshape(b,c,h*w) # b,c,hw
            w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
            w_ = w_ * (int(c)**(-0.5))
            w_ = torch.nn.functional.softmax(w_, dim=2)

            # attend to values
            v = v.reshape(b,c,h*w)
            w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
            h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
            h_ = h_.reshape(b,c,h,w)

            h_ = self.proj_out(h_)
            temp.append(h_)
        h_ = torch.cat(temp)
        return x+h_
    
class TemporalResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout, zq_ch=None, add_conv=False):
        super().__init__()
        
        self.use_3d_conv = True
        conv = nn.Conv3d
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, zq_ch, add_conv=add_conv)
        self.conv1 = conv(in_channels,
                                     out_channels,
                                     kernel_size=(3,1,1),
                                     stride=1,
                                     padding=(1,0,0))
           
        self.norm2 = Normalize(out_channels, zq_ch, add_conv=add_conv)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv(out_channels,
                                     out_channels,
                                     kernel_size=(3,1,1),
                                     stride=1,
                                     padding=(1,0,0)
        
                         )
    def forward(self, x, zq=None, num_frames=None):
        h = x
        h = self.norm1(h, zq)
        h = nonlinearity(h)
        
        h = rearrange(h, '(b t) c h w -> b c t h w', t=num_frames)
        h = self.conv1(h)
        h = rearrange(h, 'b c t h w -> (b t) c h w')

        h = self.norm2(h, zq)
        h = nonlinearity(h)
        h = self.dropout(h)

        h = rearrange(h, '(b t) c h w -> b c t h w', t=num_frames)    
        h = self.conv2(h)
        h = rearrange(h, 'b c t h w -> (b t) c h w')

        return x+h
    
class TemporalAttnBlock(nn.Module):
    def __init__(self, in_channels, zq_ch=None, add_conv=False):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, zq_ch, add_conv=add_conv)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x, num_frames, zq=None):
        bt,c,h,w = x.shape
        h_ = x        
        
        h_ = self.norm(h_, zq)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        q = rearrange(q, '(b t) c h w -> (b h w) t c', t=num_frames)
        k = rearrange(k, '(b t) c h w -> (b h w) c t', t=num_frames)
        v = rearrange(v, '(b t) c h w -> (b h w) c t', t=num_frames)
        
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = rearrange(h_, '(b h w) c t -> (b t) c h w', h=h, w=w, t=num_frames)

        h_ = self.proj_out(h_)

        return x+h_
    
class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class MOVQDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, zq_ch=None, add_conv=False, 
                 temporal=False, temporal_mid_blocks=True, temporal_kernel_size=(3,3,3), 
                 temporal_conv_padding=(1,1,1), flicker_augmentation=False, 
                 build_3D_decoder=False, temporal_attention_block=False, 
                 temporal_blocks=False, **ignorekwargs):
        super().__init__()
        
        self.build_3D_decoder=build_3D_decoder
        self.temporal_attention_block=temporal_attention_block
        self.temporal_blocks = temporal_blocks
        
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       zq_ch=zq_ch,
                                       add_conv=add_conv,
                                       temporal=temporal and temporal_mid_blocks,
                                       temporal_kernel_size=temporal_kernel_size,
                                       temporal_conv_padding=temporal_conv_padding,
                                       flicker_augmentation=flicker_augmentation,
                                        use_3d_conv=build_3D_decoder)
        self.mid.attn_1 = AttnBlock(block_in, zq_ch, add_conv=add_conv)
        
        if self.temporal_attention_block:
            self.mid.temporal_attn_1=TemporalAttnBlock(block_in, zq_ch, add_conv=add_conv)

        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       zq_ch=zq_ch,
                                       add_conv=add_conv,
                                       temporal=temporal and temporal_mid_blocks,
                                       temporal_kernel_size=temporal_kernel_size,
                                       temporal_conv_padding=temporal_conv_padding,
                                       use_3d_conv=build_3D_decoder
        )

        if self.temporal_blocks:
            self.mid.temporal_block_1 = TemporalResnetBlock(
                in_channels=block_in, 
                out_channels=block_in, 
                dropout=dropout, 
                zq_ch=zq_ch, 
                add_conv=add_conv
            )
            self.mid.temporal_block_2 = TemporalResnetBlock(
                in_channels=block_in, 
                out_channels=block_in, 
                dropout=dropout, 
                zq_ch=zq_ch, 
                add_conv=add_conv
            )
                          
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            temporal_attn = nn.ModuleList()
            temporal_block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         zq_ch=zq_ch,
                                         add_conv=add_conv,
                                         temporal=temporal,
                                         temporal_kernel_size=temporal_kernel_size,
                                         temporal_conv_padding=temporal_conv_padding,
                                         use_3d_conv=build_3D_decoder
                                       ))
                block_in = block_out
                if self.temporal_blocks:
                    temporal_block.append(
                        TemporalResnetBlock(
                            in_channels=block_in, 
                            out_channels=block_in, 
                            dropout=dropout, 
                            zq_ch=zq_ch, 
                            add_conv=add_conv
                        )
                    )

                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in, zq_ch, add_conv=add_conv))
                    if self.temporal_attention_block:
                        temporal_attn.append(TemporalAttnBlock(block_in, zq_ch, add_conv=add_conv))
                        
            up = nn.Module()
            up.block = block
            up.attn = attn
            up.temporal_attn = temporal_attn
            up.temporal_block = temporal_block
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in, zq_ch, add_conv=add_conv)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, zq, num_frames=None, aug_mask=None):
        # if temporal: assert num_frames is not None
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, zq, num_frames, aug_mask)
        h = self.mid.attn_1(h, zq)
        if self.temporal_attention_block:
            h = self.mid.temporal_attn_1(h, num_frames)
        if self.temporal_blocks:
            h = self.mid.temporal_block_1(h, zq, num_frames)
        h = self.mid.block_2(h, temb, zq, num_frames, aug_mask)
        if self.temporal_blocks:
            h = self.mid.temporal_block_2(h, zq, num_frames)
                          
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb, zq, num_frames, aug_mask)
                if self.temporal_blocks:
                    h = self.up[i_level].temporal_block[i_block](h, zq, num_frames)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
                    if self.temporal_attention_block:
                        h = self.up[i_level].temporal_attn[i_block](h, num_frames)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h, zq)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
    
class MoVQ(nn.Module):
    
    def __init__(self, generator_params):
        super().__init__()

        z_channels = generator_params["z_channels"]
        self.encoder = Encoder(**generator_params)
        self.quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.decoder = MOVQDecoder(zq_ch=z_channels, **generator_params)
        
    @torch.no_grad()
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    @torch.no_grad()
    def decode(self, quant, aug_emb=None):
        num_frames = quant.shape[0]
        decoder_input = self.post_quant_conv(quant)
        decoded = self.decoder(decoder_input, quant, num_frames, aug_emb)
        return decoded

class MoVQEncoder(nn.Module):
    
    def __init__(self, generator_params):
        super().__init__()

        z_channels = generator_params["z_channels"]
        self.encoder = Encoder(**generator_params)
        self.quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        
    @torch.no_grad()
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h
    
def get_vae(conf):
    movq = MoVQ(conf.params)
    if conf.checkpoint is not None:
        movq_state_dict = torch.load(conf.checkpoint)
        movq.load_state_dict(movq_state_dict)
    movq = freeze(movq)
    return movq

def get_vae_encoder(conf):
    movq = MoVQEncoder(conf.params)
    if conf.checkpoint is not None:
        movq_state_dict = torch.load(conf.checkpoint)
        movq.load_state_dict(movq_state_dict, strict=False)
    movq = freeze(movq)
    return movq
