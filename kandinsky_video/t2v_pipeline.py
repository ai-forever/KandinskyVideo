import warnings

from typing import List
import PIL

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from einops import repeat, rearrange
import numpy as np

from kandinsky_video.model.unet import UNet
from kandinsky_video.movq import MoVQ
from kandinsky_video.condition_encoders import T5TextConditionEncoder
from kandinsky_video.condition_processors import T5TextConditionProcessor
from kandinsky_video.model.diffusion import BaseDiffusion

SKIP_FRAMES_MEDIUM_FPS = 3
SKIP_FRAMES_HIGH_FPS = 1

MOTION_SCORES = {
    "low": 1,
    "medium": 10,
    "high": 50,
    "extreme": 100,
}

class KandinskyVideoT2VPipeline:
    
    def __init__(
        self, 
        device_map: dict,
        dtype_map: dict,
        unet: UNet,
        null_embedding: torch.Tensor,
        interpolation_unet: UNet,
        interpolation_null_embedding: torch.Tensor,
        t5_processor: T5TextConditionProcessor,
        t5_encoder: T5TextConditionEncoder,
        movq: MoVQ, video_movq:MoVQ
    ):
        self.device_map = device_map
        self.dtype_map = dtype_map
        self.to_pil = T.ToPILImage()
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.ToTensor(),
            T.Lambda(lambda img: 2. * img - 1.),
        ])
        
        self.unet = unet
        self.null_embedding = null_embedding

        self.interpolation_unet = interpolation_unet
        self.interpolation_null_embedding = interpolation_null_embedding

        self.t5_processor = t5_processor
        self.t5_encoder = t5_encoder
        self.movq = movq
        self.video_movq = video_movq
        self.base_diffusion = BaseDiffusion()
        
        self.num_frames = 12

    def encode_text(self, prompt, model_label, batch_size):
        condition_model_input, _ = self.t5_processor.encode(prompt, None)
        for input_type in condition_model_input:
            condition_model_input[input_type] = condition_model_input[input_type].unsqueeze(0).to(
                self.device_map['text_encoder']
            )
            
        with torch.cuda.amp.autocast(dtype=self.dtype_map['text_encoder']):
            context, context_mask = self.t5_encoder(condition_model_input, model_label)                
        
        bs_context = repeat(context, '1 n d -> b n d', b=batch_size)
        bs_context_mask = repeat(context_mask, '1 n -> b n', b=batch_size)
        
        return bs_context, bs_context_mask
    
    def generate_key_frame(
        self, prompt, height=512, width=512, guidance_scale=3.0
    ):
        with torch.no_grad():
            bs_context, bs_context_mask = self.encode_text(prompt, self.t5_encoder.IMAGE_GENERATION_LABEL, 1)

            with torch.cuda.amp.autocast(dtype=self.dtype_map['unet']):
                key_frame = self.base_diffusion.p_sample_loop_image(
                    self.unet, (1, 4, height // 4, width // 4), self.device_map['unet'],
                    bs_context, bs_context_mask, self.null_embedding, guidance_scale,
                )
            return self.decode_image(key_frame)
    
    def generate_base_frames(
        self, prompt, key_frame=None, 
        height=512, width=512, guidance_scale_prompt=5, guidance_weight_image=3., motion='normal', 
        noise_augmentation = 20, key_frame_guidance_scale = 3.0
    ):
        if key_frame is None:
            key_frame = self.generate_key_frame(prompt, height, width, key_frame_guidance_scale)
        
        if motion not in MOTION_SCORES.keys():
            warnings.warn(f"motion must be in {MOTION_SCORES.keys()}. set default speed to medium.")
            motion = 'medium'
        motion_score = MOTION_SCORES[motion]

        with torch.no_grad():
            
            bs_context, bs_context_mask = self.encode_text(prompt, self.t5_encoder.IMAGE_GENERATION_LABEL, self.num_frames)
            key_frame = self.encode_image(key_frame, width, height)

            with torch.cuda.amp.autocast(dtype=self.dtype_map['unet']):
                base_frames = self.base_diffusion.p_sample_loop(
                    self.unet, (self.num_frames, 4, height // 8, width // 8), self.device_map['unet'],
                    bs_context, bs_context_mask, self.null_embedding, guidance_scale_prompt,
                    key_frame=key_frame, num_frames=self.num_frames, guidance_weight_image=guidance_weight_image, motion_score = motion_score, noise_augmentation = noise_augmentation
                )
        
        return base_frames

    def interpolate_base_frames(self, base_frames, height, width, guidance_scale, skip_frames, prompt):
        
        with torch.no_grad():
            num_predicted_groups = base_frames.shape[0] - 1

            device = self.device_map['interpolation_unet']
            bs_context, bs_context_mask = self.encode_text(prompt, self.t5_encoder.IMAGE_GENERATION_LABEL, num_predicted_groups)

            with torch.cuda.amp.autocast(dtype=self.dtype_map['interpolation_unet']):

                interpolated_base_frames = self.base_diffusion.p_sample_loop_interpolation(
                    self.interpolation_unet, 
                    (num_predicted_groups, 12, height // 8, width // 8), 
                    device,
                    base_frames,
                    bs_context, 
                    bs_context_mask, 
                    self.interpolation_null_embedding, 
                    guidance_scale,
                    skip_frames=skip_frames, 
                )
                
            temporal_groups = rearrange(interpolated_base_frames, 'b (t c) h w -> b t c h w', t=3)
            temporal_groups = torch.cat([temporal_groups[i] for i in range(temporal_groups.shape[0])], axis=0)

            b, c, h, w = base_frames.shape
            video_upsampled = torch.zeros((4 * b - 3, c, h, w), device=base_frames.device)

            interpolation_indices = [i for i in range(video_upsampled.shape[0]) if i % 4 != 0]
            keyframes_indices = [i for i in range(video_upsampled.shape[0]) if i % 4 == 0]

            video_upsampled[interpolation_indices] = temporal_groups
            video_upsampled[keyframes_indices] = base_frames
            
            return video_upsampled
    
    def encode_image(self, image, width, height):
        with torch.no_grad():
            reduce_factor = max(1, min(image.size[0] / width, image.size[1] / height))
            image = image.resize((
                round(image.size[0] / reduce_factor), round(image.size[1] / reduce_factor)
            ))
            old_width, old_height = image.size
            left = (old_width - width)/2
            top = (old_height - height)/2
            right = (old_width + width)/2
            bottom = (old_height + height)/2
            image = image.crop((left, top, right, bottom))
            image = self.image_transform(image)
            with torch.cuda.amp.autocast(dtype=self.dtype_map['movq']):
                image = image.unsqueeze(0).to(device=self.device_map['movq'], dtype=self.dtype_map['movq'])
                image = self.movq.encode(image)[0]
            return image
    
    def decode_image(self, image):        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype_map['movq']):
                image = self.movq.decode(image)
                image = torch.clip((image + 1.) / 2., 0., 1.)
                image = 255. * image.permute(0, 2, 3, 1).cpu().numpy()[0]
        
        return PIL.Image.fromarray(image.astype(np.uint8)) 
        
    def decode_video(self, video):        
        pil_video = []
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype_map['movq']):
                video = torch.cat([self.video_movq.decode(frame) for frame in video.chunk(video.shape[0] // 4)])
                video = torch.clip((video + 1.) / 2., 0., 1.)
                for video_chunk in video.chunk(1):
                    pil_video += [self.to_pil(frame) for frame in video_chunk]
        return pil_video
    
    def __call__(
        self, 
        prompt: str,
        negative_prompt: str = None,
        image: PIL.Image.Image = None,
        width: int = 512,
        height: int = 512,
        fps: str = 'low',
        motion: str = 'normal',
        key_frame_guidance_scale: float = 5.0,
        guidance_weight_prompt: float = 5.0,
        guidance_weight_image: float = 2.0,
        interpolation_guidance_scale: float = 0.5,
        noise_augmentation = 20,
    ) -> List[PIL.Image.Image]:
                
        video = self.generate_base_frames(
            prompt, image, height, width, guidance_weight_prompt, guidance_weight_image, motion, noise_augmentation, key_frame_guidance_scale
        )

        if fps in ['medium', 'high']:
            video = self.interpolate_base_frames(
                video, height, width, interpolation_guidance_scale, SKIP_FRAMES_MEDIUM_FPS, prompt
            )

            if fps == 'high':
                video = self.interpolate_base_frames(
                    video, height, width, interpolation_guidance_scale, SKIP_FRAMES_HIGH_FPS, prompt
                )

        pil_video = self.decode_video(video)
        return pil_video
