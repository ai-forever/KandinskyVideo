# Kandinsky Video — a new text-to-video generation model 
## SoTA quality among open-source solutions

This repository is the official implementation of Kandinsky Video model


[Paper](https://arxiv.org/abs/2311.13073) | [Project](https://ai-forever.github.io/kandinsky-video/) | [![Hugging Face Spaces](https://img.shields.io/badge/🤗-Huggingface-yello.svg)](https://huggingface.co/ai-forever/KandinskyVideo) | [Telegram-bot](https://t.me/video_kandinsky_bot) | [Habr post](https://habr.com/ru/companies/sberbank/articles/775554/)


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

Check our jupyter notebooks with examples in `./examples` folder
### 1. text2video

```python
from video_kandinsky3 import get_T2V_pipeline

t2v_pipe = get_T2V_pipeline('cuda', fp16=True)

pfps = 'medium' # ['low', 'medium', 'high']
video = t2v_pipe(
    'a red car is drifting on the mountain road, close view, fast movement',
    width=640, height=384, fps=fps
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


# Authors

+ Vladimir Arkhipkin: [Github](https://github.com/oriBetelgeuse), [Google Scholar](https://scholar.google.com/citations?user=D-Ko0oAAAAAJ&hl=ru)
+ Zein Shaheen: [Github](https://github.com/zeinsh), [Google Scholar](https://scholar.google.ru/citations?user=bxlgMxMAAAAJ&hl=en)
+ Viacheslav Vasilev: [Github](https://github.com/vivasilev), [Google Scholar](https://scholar.google.com/citations?user=redAz-kAAAAJ&hl=ru&oi=sra)
+ Igor Pavlov: [Github](https://github.com/boomb0om)
+ Elizaveta Dakhova: [Github](https://github.com/LizaDakhova)
+ Anastasia Lysenko: [Github](https://github.com/LysenkoAnastasia)
+ Sergey Markov
+ Denis Dimitrov: [Github](https://github.com/denndimitrov), [Google Scholar](https://scholar.google.com/citations?user=3JSIJpYAAAAJ&hl=ru&oi=ao)
+ Andrey Kuznetsov: [Github](https://github.com/kuznetsoffandrey), [Google Scholar](https://scholar.google.com/citations?user=q0lIfCEAAAAJ&hl=ru)


## BibTeX
If you use our work in your research, please cite our publication:
```
@article{arkhipkin2023fusionframes,
  title     = {FusionFrames: Efficient Architectural Aspects for Text-to-Video Generation Pipeline},
  author    = {Arkhipkin, Vladimir and Shaheen, Zein and Vasilev, Viacheslav and Dakhova, Elizaveta and Kuznetsov, Andrey and Dimitrov, Denis},
  journal   = {arXiv preprint arXiv:2311.13073},
  year      = {2023}, 
}
```
