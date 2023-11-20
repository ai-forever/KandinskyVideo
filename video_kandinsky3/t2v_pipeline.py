from typing import Optional, Union, List
import PIL
import io
import os
import math
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision.transforms as T
from torch import einsum
from einops import repeat, rearrange

from video_kandinsky3.model.unet import UNet
from video_kandinsky3.movq import MoVQ
from video_kandinsky3.condition_encoders import T5TextConditionEncoder
from video_kandinsky3.condition_processors import T5TextConditionProcessor
from video_kandinsky3.model.diffusion import BaseDiffusion, get_named_beta_schedule


class VideoKandinsky3T2VPipeline:
    
    def __init__(
        self, 
        device: Union[str, torch.device], 
        unet: UNet,
        null_embedding: torch.Tensor,
        interpolation_unet: UNet,
        interpolation_null_embedding: torch.Tensor,
        t5_processor: T5TextConditionProcessor,
        t5_encoder: T5TextConditionEncoder,
        movq: MoVQ,
        fp16: bool = True
    ):
        self.device = device
        self.fp16 = fp16
        self.to_pil = T.ToPILImage()
        
        self.unet = unet
        self.null_embedding = null_embedding

        self.interpolation_unet = interpolation_unet
        self.interpolation_null_embedding = interpolation_null_embedding

        self.t5_processor = t5_processor
        self.t5_encoder = t5_encoder
        self.movq = movq

    def _reshape_temporal_groups(self, temporal_groups, video):
        temporal_groups = rearrange(temporal_groups, 'b (t c) h w -> b t c h w', t=3)
        temporal_groups = torch.cat([temporal_groups[i] for i in range(temporal_groups.shape[0])], axis=0)

        b, c, h, w = video.shape
        video_upsampled = torch.zeros((4 * b - 3, c, h, w), device=self.device)

        interpolation_indices = [i for i in range(video_upsampled.shape[0]) if i % 4 != 0]
        keyframes_indices = [i for i in range(video_upsampled.shape[0]) if i % 4 == 0]

        video_upsampled[interpolation_indices] = temporal_groups
        video_upsampled[keyframes_indices] = video
        return video_upsampled

    def generate_base_frames(
            self, base_diffusion, height, width, guidance_scale, condition_model_input,
            negative_condition_model_input=None
    ):
        context, context_mask = self.t5_encoder(condition_model_input)
        if negative_condition_model_input is not None:
            negative_context, negative_context_mask = self.t5_encoder(negative_condition_model_input)
        else:
            negative_context, negative_context_mask = None, None

        bs_context = repeat(context, '1 n d -> b n d', b=self.unet.num_frames)
        bs_context_mask = repeat(context_mask, '1 n -> b n', b=self.unet.num_frames)
        if negative_context is not None:
            bs_negative_context = repeat(negative_context, '1 n d -> b n d', b=self.unet.num_frames)
            bs_negative_context_mask = repeat(negative_context_mask, '1 n -> b n', b=self.unet.num_frames)
        else:
            bs_negative_context, bs_negative_context_mask = None, None

        video_len = 180
        temporal_positions = torch.arange(0, video_len, video_len // self.unet.num_frames, device='cuda:0')
        base_frames = base_diffusion.p_sample_loop(
            self.unet, (self.unet.num_frames, 4, height // 8, width // 8), self.device,
            bs_context, bs_context_mask, self.null_embedding, guidance_scale,
            temporal_positions=temporal_positions,
            negative_context=bs_negative_context, negative_context_mask=bs_negative_context_mask
        )
        return base_frames

    def interpolate_base_frames(self, base_diffusion, base_frames, height, width, guidance_scale, skip_frames):
        num_temporal_groups = base_frames.shape[0] - 1
        left_base_frames, right_base_frames = base_frames[:-1], base_frames[1:]

        bs_context = torch.zeros([num_temporal_groups, 2, 4096], device=self.device)
        bs_context_mask = torch.zeros([num_temporal_groups, 2], device=self.device)
        bs_context[:, 0] = self.interpolation_null_embedding
        bs_context_mask[:, 0] = 1

        skip_frames = skip_frames * torch.ones(size=(num_temporal_groups,), dtype=torch.int32, device=self.device)

        interpolated_base_frames = base_diffusion.p_sample_loop(
            self.interpolation_unet, (num_temporal_groups, 12, height // 8, width // 8), self.device,
            bs_context, bs_context_mask, self.interpolation_null_embedding, guidance_scale,
            base_frames=(left_base_frames, right_base_frames), num_temporal_groups=num_temporal_groups,
            skip_frames=skip_frames, v_predication=True

        )
        return interpolated_base_frames
        
    def __call__(
        self, 
        text: str,
        negative_text: str = None,
        width: int = 512,
        height: int = 512,
        fps: str = 'low',
        guidance_scale: float = 5.0,
        interpolation_guidance_scale: float = 0.25,
        steps: int = 50
    ) -> List[PIL.Image.Image]:

        betas = get_named_beta_schedule('cosine', steps)
        base_diffusion = BaseDiffusion(betas, 0.98)
        
        condition_model_input, negative_condition_model_input = self.t5_processor.encode(text, negative_text)
        for key in condition_model_input:
            for input_type in condition_model_input[key]:
                condition_model_input[key][input_type] = condition_model_input[key][input_type].unsqueeze(0).to(self.device)

        if negative_condition_model_input is not None:
            for key in negative_condition_model_input:
                for input_type in negative_condition_model_input[key]:
                    negative_condition_model_input[key][input_type] = negative_condition_model_input[key][input_type].unsqueeze(0).to(self.device)

        pil_video = []
        with torch.cuda.amp.autocast(enabled=self.fp16):
            with torch.no_grad():
                video = self.generate_base_frames(
                    base_diffusion, height, width, guidance_scale, condition_model_input, negative_condition_model_input
                )
                if fps in ['medium', 'high']:
                    temporal_groups = self.interpolate_base_frames(
                        base_diffusion, video, height, width, interpolation_guidance_scale, 3
                    )
                    video = self._reshape_temporal_groups(temporal_groups, video)
                    if fps == 'high':
                        temporal_groups = self.interpolate_base_frames(
                            base_diffusion, video, height, width, interpolation_guidance_scale, 1
                        )
                        video = self._reshape_temporal_groups(temporal_groups, video)

                video = torch.cat([self.movq.decode(frame) for frame in video.chunk(video.shape[0] // 4)])
                video = torch.clip((video + 1.) / 2., 0., 1.)
                for video_chunk in video.chunk(1):
                    pil_video += [self.to_pil(frame) for frame in video_chunk]
        return pil_video
