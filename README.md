# Kandinsky Video — a new text-to-video generation model 
## SoTA quality among open-source solutions

This repository is the official implementation of Kandinsky Video model

</br>
Vladimir Arkhipkin,
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


## How to use

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


## Results


<table class="center">
<tr>
  <td><img src="__assets__/results/A car moving on the road from the sea to the mountains.gif" raw=true></td>
  <td><img src="__assets__/results/A red car drifting, 4k video.gif"></td>
  <td><img src="__assets__/results/chemistry laboratory, chemical explosion, 4k.gif"></td>
  <td><img src="__assets__/results/Erupting volcano_ raw power, molten lava, and the forces of the Earth.gif"></td>
</tr>
<tr>
  <td width=25% align="center">"A car moving on the road from the sea to the mountains"</td>
  <td width=25% align="center">"A red car drifting, 4k video"</td>
  <td width=25% align="center">"Chemistry laboratory, chemical explosion, 4k"</td>
  <td width=25% align="center">"Erupting volcano raw power, molten lava, and the forces of the Earth"</td>
</tr>

<tr>
  <td><img src="__assets__/results/luminescent jellyfish swims underwater, neon, 4k.gif" raw=true></td>
  <td><img src="__assets__/results/Majestic waterfalls in a lush rainforest_ power, mist, and biodiversity.gif"></td>
  <td><img src="__assets__/results/white ghost flies through a night clearing, 4k.gif"></td>
  <td><img src="__assets__/results/Wildlife migration_ herds on the move, crossing landscapes in harmony.gif"></td>
</tr>
<tr>
  <td width=25% align="center">"Luminescent jellyfish swims underwater, neon, 4k"</td>
  <td width=25% align="center">"Majestic waterfalls in a lush rainforest power, mist, and biodiversity"</td>
  <td width=25% align="center">"White ghost flies through a night clearing, 4k"</td>
  <td width=25% align="center">"Wildlife migration herds on the move, crossing landscapes in harmony"</td>
</tr>

<tr>
  <td><img src="__assets__/results/Majestic humpback whale breaching_ power, grace, and ocean spectacle.gif" raw=true></td>
  <td><img src="__assets__/results/Evoke the sense of wonder in a time-lapse journey through changing seasons..gif"></td>
  <td><img src="__assets__/results/Explore the fascinating world of underwater creatures in a visually stunning sequence.gif"></td>
  <td><img src="__assets__/results/Polar ice caps_ the pristine wilderness of the Arctic and Antarctic.gif"></td>
</tr>
<tr>
  <td width=25% align="center">"Majestic humpback whale breaching power, grace, and ocean spectacle"</td>
  <td width=25% align="center">"Evoke the sense of wonder in a time-lapse journey through changing seasons"</td>
  <td width=25% align="center">"Explore the fascinating world of underwater creatures in a visually stunning sequence"</td>
  <td width=25% align="center">"Polar ice caps the pristine wilderness of the Arctic and Antarctic"</td>
</tr>


<tr>
  <td><img src="__assets__/results/Rolling waves on a sandy beach_ relaxation, rhythm, and coastal beauty.gif" raw=true></td>
  <td><img src="__assets__/results/Sloth in slow motion_ deliberate movements, relaxation, and arboreal life.gif"></td>
  <td><img src="__assets__/results/Time-lapse of a flower blooming_ growth, beauty, and the passage of time..gif"></td>
  <td><img src="__assets__/results/Craft a heartwarming narrative showcasing the bond between a human and their loyal pet companion..gif"></td>
</tr>
<tr>
  <td width=25% align="center">"Rolling waves on a sandy beach relaxation, rhythm, and coastal beauty"</td>
  <td width=25% align="center">"Sloth in slow motion deliberate movements, relaxation, and arboreal life"</td>
  <td width=25% align="center">"Time-lapse of a flower blooming growth, beauty, and the passage of time"</td>
  <td width=25% align="center">"Craft a heartwarming narrative showcasing the bond between a human and their loyal pet companion"</td>
</tr>


</table>



## BibTeX
If you use our work in your research, please cite our publication:
```

```
