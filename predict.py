# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import imageio
import numpy as np
from cog import BasePredictor, Input, Path
from video_kandinsky3 import *


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        model_weights_dir = "model_weights"
        self.t2v_pipe = get_T2V_pipeline(
            "cuda:0",
            fp16=True,
            cache_dir=model_weights_dir,
            unet_path=f"{model_weights_dir}/weights/kandinsky_video.pt",
            interpolation_unet_path=f"{model_weights_dir}/weights/kandinsky_video_interpolation.pt",
            movq_path=f"{model_weights_dir}/weights/movq.pt",
            text_encode_path="google_flan_ul2_weights",  # pre-loaded from google/flan-ul2
        )

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt.",
            default="a red car is drifting on the mountain road, close view, fast movement",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output.",
            default=None,
        ),
        width: int = Input(
            description="Width of output video. Lower the setting if out of memory.",
            default=640,
        ),
        height: int = Input(
            description="Height of output video. Lower the setting if out of memory.",
            default=384,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", default=5.0
        ),
        interpolation_guidance_scale: float = Input(
            description="Scale for interpolation guidance", default=0.25
        ),
        interpolation_level: str = Input(
            choices=["low", "medium", "high"],
            default="low",
        ),
        fps: int = Input(
            description="fps for the output video.",
            default=10,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        video = self.t2v_pipe(
            text=prompt,
            negative_text=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            interpolation_guidance_scale=interpolation_guidance_scale,
            steps=num_inference_steps,
            fps=interpolation_level,
        )
        output_path = "/tmp/output.mp4"
        imageio.mimsave(output_path, [np.array(im) for im in video], fps=fps)

        return Path(output_path)
