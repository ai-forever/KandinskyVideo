import os
import json
from tqdm import tqdm
from typing import Optional, Union

import torch
from huggingface_hub import hf_hub_download

from kandinsky_video.model.unet import UNet
from kandinsky_video.model.unet_interpolation import UNet as UNetInterpolation
from kandinsky_video.movq import MoVQ
from kandinsky_video.condition_encoders import T5TextConditionEncoder
from kandinsky_video.condition_processors import T5TextConditionProcessor

from .t2v_pipeline import KandinskyVideoT2VPipeline

REPO_ID="ai-forever/KandinskyVideo_1_1"

def get_T2V_unet(
        cache_dir: str,
        device: Union[str, torch.device],
        configs: Optional[dict] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
) -> (UNet, Optional[dict], Optional[torch.Tensor]):
    
    weights_path = hf_hub_download(
        repo_id=REPO_ID, filename='t2v.pt', cache_dir=cache_dir
    )

    unet = UNet(**configs)

    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    null_embedding = state_dict['null_embedding']
    unet.load_state_dict(state_dict['unet'])
    unet.to(device=device, dtype=dtype).eval()
    
    return unet, null_embedding, None

def get_interpolation_unet(
        cache_dir: str,
        device: Union[str, torch.device],
        configs: Optional[dict] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
) -> (UNet, Optional[dict], Optional[torch.Tensor]):
    
    weights_path = hf_hub_download(
        repo_id=REPO_ID, filename='interpolation.pt', cache_dir=cache_dir
    )

    unet = UNetInterpolation(**configs)

    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    null_embedding = state_dict['null_embedding']
    unet.load_state_dict(state_dict['unet'])

    unet.to(device=device, dtype=dtype).eval()
    return unet, null_embedding, None

def get_T5encoder(
    cache_dir: str,
    device: Union[str, torch.device],
    weights_path: str,
    tokens_length: int = 128,
    context_dim: int = 4096,
    dtype: Union[str, torch.dtype] = torch.float32,
    low_cpu_mem_usage: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> (T5TextConditionProcessor, T5TextConditionEncoder):
    
    t5_projections_path = hf_hub_download(
        repo_id=REPO_ID, filename='t5_projections.pt', cache_dir=cache_dir
    )

    processor = T5TextConditionProcessor(tokens_length, weights_path)
    condition_encoder = T5TextConditionEncoder(
        weights_path, t5_projections_path, context_dim, low_cpu_mem_usage=low_cpu_mem_usage, 
        device=device, dtype=dtype, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
    )

    return processor, condition_encoder


def get_movq(
    cache_dir: str,
    device: Union[str, torch.device],
    configs: dict = None,
    dtype: Union[str, torch.dtype] = torch.float32,
) -> MoVQ:
    
    weights_path = hf_hub_download(
        repo_id=REPO_ID, filename='movq.pt', cache_dir=cache_dir
    )

    movq = MoVQ(configs)

    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    movq.load_state_dict(state_dict)

    movq.to(device=device, dtype=dtype).eval()
    return movq

def get_video_movq(
    cache_dir: str,
    device: Union[str, torch.device],
    configs: dict = None,
    dtype: Union[str, torch.dtype] = torch.float32,
) -> MoVQ:

    weights_path = hf_hub_download(
        repo_id=REPO_ID, filename='video_movq.pt', cache_dir=cache_dir
    )

    movq = MoVQ(configs)

    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    movq.load_state_dict(state_dict)

    movq.to(device=device, dtype=dtype).eval()
    return movq

def get_T2V_pipeline(
    device_map: Union[str, torch.device, dict],
    cache_dir: str = '/tmp/kandinsky_video/',
) -> KandinskyVideoT2VPipeline:

    configs_path = hf_hub_download(
        repo_id=REPO_ID, filename='configs.json', cache_dir=cache_dir
    )
    configs=json.load(open(configs_path))
    
    if not isinstance(device_map, dict):
        device_map = {
            'unet': device_map, 'interpolation_unet': device_map, 'text_encoder': device_map, 'movq': device_map
        }
    
    dtype_map = {
        'unet': torch.float16,
        'interpolation_unet': torch.float16,
        'text_encoder': torch.float32,
        'movq': torch.float32,
    }

    unet, null_embedding, _ = get_T2V_unet(
        cache_dir,
        device_map['unet'], 
        dtype=dtype_map['unet'], 
        **configs['t2v']
    )

    interpolation_unet, interpolation_null_embedding, _ = get_interpolation_unet(
        cache_dir,
        device_map['interpolation_unet'], 
        dtype=dtype_map['interpolation_unet'], 
        **configs['interpolation']
    )
        
    processor, condition_encoder = get_T5encoder(
        cache_dir,
        device_map['text_encoder'],  
        dtype=dtype_map['text_encoder'],
        **configs['text_encoder']
    )
    movq = get_movq(
        cache_dir,
        device_map['movq'], 
        dtype=dtype_map['movq'], 
        **configs["image_movq"]
    )
    video_movq = get_video_movq(
        cache_dir,
        device_map['movq'], 
        dtype=dtype_map['movq'], 
        **configs["video_movq"]
    )
    return KandinskyVideoT2VPipeline(
        device_map, dtype_map, unet, null_embedding, interpolation_unet, interpolation_null_embedding,
        processor, condition_encoder, movq, video_movq
    )
