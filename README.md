# KandinskyVideo — a new text-to-video generation model 
## SoTA quality among open-source solutions

**Architecture details:**

+ Text encoder (Flan-UL2) - 8.6B
+ Latent Diffusion U-Net3D - 4.0B
+ MoVQ encoder/decoder - 256M

## How to use:

Check our jupyter notebooks with examples in `./notebooks` folder
### 1. text2video

```python
from video_kandinsky3 import get_T2V_unet, get_interpolation_unet, get_T5encoder, get_movq, VideoKandinsky3T2VPipeline

unet, null_embedding, projections_state_dict = get_T2V_unet('cuda', kandinsky_video.pt, fp16=True)
interpolation_unet, interpolation_null_embedding = get_interpolation_unet('cuda', kandinsky_video_interpolation.pt, fp16=True)
processor, condition_encoders = get_T5encoder('cuda', 'google/flan-ul2', projections_state_dict)
movq = get_movq('cuda', 'movq.pt', fp16=True)

pipeline = VideoKandinsky3T2VPipeline(
    'cuda', unet, null_embedding, interpolation_unet, interpolation_null_embedding, processor,
    condition_encoders, movq, True
)
video = pipeline(
    'a red car is drifting on the mountain road, close view, fast movement',
    width=640, height=384, fps='medium'
)
