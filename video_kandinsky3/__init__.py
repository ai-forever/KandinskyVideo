from typing import Optional, Union, List
import PIL
import os

import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download

import torch
import omegaconf
from omegaconf import OmegaConf
from video_kandinsky3.model.unet import UNet
from video_kandinsky3.movq import MoVQ
from video_kandinsky3.condition_encoders import T5TextConditionEncoder
from video_kandinsky3.condition_processors import T5TextConditionProcessor
from video_kandinsky3.model.diffusion import BaseDiffusion, get_named_beta_schedule

from .t2v_pipeline import VideoKandinsky3T2VPipeline


def get_T2V_unet(
        device: Union[str, torch.device],
        weights_path: Optional[str] = None,
        fp16: bool = False
) -> (UNet, Optional[dict], Optional[torch.Tensor]):
    unet = UNet(
        model_channels=384,
        num_channels=4,
        out_channels=4,
        init_channels=192,
        time_embed_dim=1536,
        context_dim=4096,
        groups=32,
        head_dim=64,
        expansion_ratio=4,
        compression_ratio=2,
        dim_mult=(1, 2, 4, 8),
        num_blocks=(3, 3, 3, 3),
        add_cross_attention=(False, True, True, True),
        add_self_attention=(False, True, True, True),
        num_frames=15
    )

    # load weights
    null_embedding = None
    projections_state_dict = None
    if weights_path:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        projections_state_dict = state_dict['projections']
        null_embedding = state_dict['null_embedding']
        unet.load_state_dict(state_dict['unet'])

    unet.eval().to(device)

    if fp16:
        unet = unet.half()

    return unet, null_embedding, projections_state_dict


def get_interpolation_unet(
        device: Union[str, torch.device],
        weights_path: Optional[str] = None,
        fp16: bool = False
) -> (UNet, Optional[dict], Optional[torch.Tensor]):
    unet = UNet(
        model_channels=384,
        num_channels=20,
        out_channels=12,
        init_channels=192,
        time_embed_dim=1536,
        context_dim=4096,
        groups=32,
        head_dim=64,
        expansion_ratio=4,
        compression_ratio=2,
        dim_mult=(1, 2, 4, 8),
        num_blocks=(3, 3, 3, 3),
        add_cross_attention=(False, True, True, True),
        add_self_attention=(False, True, True, True),
        interpolation=True
    )

    # load weights
    null_embedding = None
    if weights_path:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        null_embedding = state_dict['null_embedding']
        unet.load_state_dict(state_dict['unet'])

    unet.eval().to(device)

    if fp16:
        unet = unet.half()

    return unet, null_embedding


def get_T5encoder(
        device: Union[str, torch.device],
        weights_path: str,
        projections_state_dict: Optional[dict] = None,
        fp16: bool = True,
        low_cpu_mem_usage: bool = True,
        device_map: Optional[str] = None
) -> (UNet, Optional[dict], Optional[torch.Tensor]):
    model_names = {'t5': weights_path}
    tokens_length = {'t5': 128}
    context_dim = 4096
    model_dims = {'t5': 4096}
    processor = T5TextConditionProcessor(tokens_length, model_names)
    condition_encoders = T5TextConditionEncoder(
        model_names, context_dim, model_dims, low_cpu_mem_usage=low_cpu_mem_usage, device_map=device_map
    )

    if projections_state_dict:
        condition_encoders.projections.load_state_dict(projections_state_dict)

    condition_encoders = condition_encoders.eval().to(device)
    return processor, condition_encoders


def get_movq(
        device: Union[str, torch.device],
        weights_path: str,
        fp16: bool = False
) -> (UNet, Optional[dict], Optional[torch.Tensor]):
    generator_config = {
        'double_z': False,
        'z_channels': 4,
        'resolution': 256,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 256,
        'ch_mult': [1, 2, 2, 4],
        'num_res_blocks': 2,
        'attn_resolutions': [32],
        'dropout': 0.0
    }
    movq = MoVQ(generator_config)
    movq.load_state_dict(torch.load(weights_path))
    movq = movq.eval().to(device)

    if fp16:
        movq = movq.half()
    return movq


def get_T2V_pipeline(
    device: Union[str, torch.device],
    fp16: bool = False,
    cache_dir: str = '/tmp/kandinsky_video/',
    unet_path: str = None,
    interpolation_unet_path: str = None,
    text_encode_path: str = None,
    movq_path: str = None,
) -> VideoKandinsky3T2VPipeline:
    if unet_path is None:
        unet_path = hf_hub_download(
            repo_id="ai-forever/KandinskyVideo", filename='weights/kandinsky_video.pt', cache_dir=cache_dir
        )
    if interpolation_unet_path is None:
        interpolation_unet_path = hf_hub_download(
            repo_id="ai-forever/KandinskyVideo", filename='weights/kandinsky_video_interpolation.pt', cache_dir=cache_dir
        )
    if text_encode_path is None:
        text_encode_path = 'google/flan-ul2'
    if movq_path is None:
        movq_path = hf_hub_download(
            repo_id="ai-forever/KandinskyVideo", filename='weights/movq.pt', cache_dir=cache_dir
        )

    unet, null_embedding, projections_state_dict = get_T2V_unet(device, unet_path, fp16=fp16)
    interpolation_unet, interpolation_null_embedding = get_interpolation_unet(device, interpolation_unet_path, fp16=fp16)
    processor, condition_encoders = get_T5encoder(device, text_encode_path, projections_state_dict, fp16=fp16)
    movq = get_movq(device, movq_path, fp16=fp16)
    return VideoKandinsky3T2VPipeline(
        device, unet, null_embedding, interpolation_unet, interpolation_null_embedding,
        processor, condition_encoders, movq, fp16=fp16
    )
