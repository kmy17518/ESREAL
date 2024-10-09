import io
import time

import requests
import torch
from huggingface_hub import hf_hub_download
from openai import OpenAI

from .lpw_stable_diffusion_xl import SDXLLongPromptWeightingPipeline
from .registry import registry


class DallEPipeline:
    def __init__(self):
        self.client = OpenAI()

    def generate_image(self, prompt: str, size: str = "1024x1024", quality: str = "standard", n: int = 1):
        """
        https://platform.openai.com/docs/guides/images/introduction
        """
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
        )
        images = []
        for image in response.data:
            url = image.url
            image = requests.get(url).content
            images.append(image)
        return images

    def __call__(self, prompt, num_images_per_prompt, *args, **kwargs):
        return self.generate_image(prompt, n=num_images_per_prompt)


class SDXLTurboPipeline:
    def __init__(self):
        self.pipeline = SDXLLongPromptWeightingPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            safety_checker=None,
        ).to(registry.device)
        self.pipeline.set_progress_bar_config(disable=True)

    def __call__(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)


class HyperSDXLPipeline:
    def __init__(self):
        self.pipeline = SDXLLongPromptWeightingPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            safety_checker=None,
        ).to(registry.device)
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-SDXL-8steps-CFG-lora.safetensors"
        self.pipeline.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        self.pipeline.fuse_lora()
        self.pipeline.set_progress_bar_config(disable=True)

        from diffusers import DDIMScheduler

        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config, timestep_spacing="trailing")

    def __call__(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)
