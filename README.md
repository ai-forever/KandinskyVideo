# Kandinsky Video — a new text-to-video generation model 
## SoTA quality among open-source solutions

This repository is the official implementation of Kandinsky Video model

</br>
**Authors** Vladimir Arkhipkin,
Zein Shaheen,
Viacheslav Vasilev,
Igor Pavlov,
Elizaveta Dakhova,
Anastasia Lysenko,
Sergey Markov,
Denis Dimitrov,
Andrey Kuznetsov
</br>

Paper | [Project](https://ai-forever.github.io/kandinsky-video/) | Hugging Face Spaces | Telegram-bot | Habr post


<p align="center">
<img src="__assets__/title.JPG" width="800px"/>  
<br>
<em>Kandinsky Video is a text-to-video generation model, which is based on the FusionFrames architecture, consisting of two main stages: keyframe generation and interpolation. Our approach for temporal conditioning allows us to generate videos with high-quality appearance, smoothness and dynamics.</em>
</p>


## Pipeline

<p align="center">
<img src="__assets__/pipeline.jpg" width="800px"/>  
<br>
<em>The encoded text prompt enters the U-Net keyframe generation model with temporal layers or blocks, and then the sampled latent keyframes are  sent to the latent interpolation model in such a way as to predict three interpolation frames between two keyframes. A temporal MoVQ-GAN decoder is used to get the final video result.</em>
</p>


**Architecture details**

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
```


## BibTeX
If you use our work in your research, please cite our publication:
```

```
