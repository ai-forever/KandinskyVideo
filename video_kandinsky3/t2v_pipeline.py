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
from einops import repeat

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
        self.t5_processor = t5_processor
        self.t5_encoder = t5_encoder
        self.movq = movq
        
    def __call__(
        self, 
        text: str,
        negative_text: str = None,
        width: int = 512,
        height: int = 512,
        guidance_scale: float = 5.0,
        steps: int = 50
    ) -> List[PIL.Image.Image]:

        betas = get_named_beta_schedule('cosine', steps)
        base_diffusion = BaseDiffusion(betas, 0.99)
        
        condition_model_input, negative_condition_model_input = self.t5_processor.encode(text, negative_text)
        for key in condition_model_input:
            for input_type in condition_model_input[key]:
                condition_model_input[key][input_type] = condition_model_input[key][input_type].unsqueeze(0).to(self.device)

        if negative_condition_model_input is not None:
            for key in negative_condition_model_input:
                for input_type in negative_condition_model_input[key]:
                    negative_condition_model_input[key][input_type] = negative_condition_model_input[key][input_type].unsqueeze(0).to(self.device)

        num_frames = self.unet.num_frames
        pil_images = []
        with torch.cuda.amp.autocast(enabled=self.fp16):
            with torch.no_grad():
                context, context_mask = self.t5_encoder(condition_model_input)
                if negative_condition_model_input is not None:
                    negative_context, negative_context_mask = self.t5_encoder(negative_condition_model_input)
                else:
                    negative_context, negative_context_mask = None, None

                bs_context = repeat(context, '1 n d -> b n d', b=num_frames)
                bs_context_mask = repeat(context_mask, '1 n -> b n', b=num_frames)
                if negative_context is not None:
                    bs_negative_context = repeat(negative_context, '1 n d -> b n d', b=num_frames)
                    bs_negative_context_mask = repeat(negative_context_mask, '1 n -> b n', b=num_frames)
                else:
                    bs_negative_context, bs_negative_context_mask = None, None

                video_len = 180
                temporal_positions = torch.arange(0, video_len, video_len // num_frames, device='cuda:0')
                base_frames = base_diffusion.p_sample_loop(
                    self.unet, (num_frames, 4, height//8, width//8), self.device,
                    bs_context, bs_context_mask, temporal_positions, self.null_embedding, guidance_scale,
                    negative_context=bs_negative_context, negative_context_mask=bs_negative_context_mask
                )

                base_frames = torch.cat([self.movq.decode(base_frame) for base_frame in base_frames.chunk(2)])
                base_frames = torch.clip((base_frames + 1.) / 2., 0., 1.)
                for base_frames_chunk in base_frames.chunk(1):
                    pil_images += [self.to_pil(base_frame) for base_frame in base_frames_chunk]

        return pil_images
