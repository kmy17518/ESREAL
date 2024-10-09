from typing import List, Tuple

import numpy as np
import torch
from loguru import logger
from PIL import Image

from .gdino_registry import create_gdino
from .registry import registry
from .reward_fn import RewardCalculator
from .t2i_pipelines import DallEPipeline, HyperSDXLPipeline, SDXLTurboPipeline

GDINO_CONFIG_PATH = "/app/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GDINO_CHECKPOINT_PATH = "/app/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"


@registry.register("reward_pipeline")
class T2IRewardPipeline:
    def __init__(self):
        # self.pipeline = SDXLTurboPipeline()
        self.pipeline = HyperSDXLPipeline()

        # self.pipeline.unet = torch.compile(self.pipeline.unet, mode="reduce-overhead")

        self.num_images_per_prompt = 4
        logger.info(f"[*] num_images_per_prompt: {self.num_images_per_prompt}")

        self.gdino_model = create_gdino(
            config_path=GDINO_CONFIG_PATH,
            checkpoint_path=GDINO_CHECKPOINT_PATH,
            device=registry.device,
        )

        self.reward_calculator = RewardCalculator(
            gdino_model=self.gdino_model,
            device=registry.device,
        )

    @torch.no_grad()
    def __call__(
        self, input_image: List[Image.Image], prompt: List[str], tokenized_prompt: List[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = len(prompt)
        assert batch_size == 1

        try:
            generated_images = self.pipeline(
                prompt[0],
                num_images_per_prompt=self.num_images_per_prompt,
                num_inference_steps=8,
                guidance_scale=8.0,
            ).images

            rec_reward_bi = [[None] * self.num_images_per_prompt]
            obj_penalty_bi = [[None] * self.num_images_per_prompt]
            att_penalty_bi = [[None] * self.num_images_per_prompt]
            rel_penalty_bi = [[None] * self.num_images_per_prompt]
            pos_penalty_bi = [[None] * self.num_images_per_prompt]

            for i in range(self.num_images_per_prompt):
                rec_reward, obj_penalty, att_penalty, rel_penalty, pos_penalty = self.reward_calculator(
                    prompt=prompt[0],
                    tokenized_prompt=tokenized_prompt[0],
                    image=input_image[0],
                    model_image=generated_images[i],
                )
                rec_reward_bi[0][i] = rec_reward
                obj_penalty_bi[0][i] = obj_penalty
                att_penalty_bi[0][i] = att_penalty
                rel_penalty_bi[0][i] = rel_penalty
                pos_penalty_bi[0][i] = pos_penalty

            rec_reward_bi = np.array(rec_reward_bi)  # (1, P, T)
            obj_penalty_bi = np.array(obj_penalty_bi)  # (1, P, T)
            att_penalty_bi = np.array(att_penalty_bi)  # (1, P, T)
            rel_penalty_bi = np.array(rel_penalty_bi)  # (1, P, T)
            pos_penalty_bi = np.array(pos_penalty_bi)  # (1, P, T)

            mean_rec_reward = rec_reward_bi.mean(axis=1).astype(np.float32)  # (1, T)
            mean_obj_penalty = obj_penalty_bi.mean(axis=1).astype(np.float32)  # (1, T)
            mean_att_penalty = att_penalty_bi.mean(axis=1).astype(np.float32)  # (1, T)
            mean_rel_penalty = rel_penalty_bi.mean(axis=1).astype(np.float32)  # (1, T)
            mean_pos_penalty = pos_penalty_bi.mean(axis=1).astype(np.float32)  # (1, T)

            assert len(mean_rec_reward.shape) == 2 and mean_rec_reward.shape[0] == 1

            return mean_rec_reward, mean_obj_penalty, mean_att_penalty, mean_rel_penalty, mean_pos_penalty

        except Exception as e:
            logger.error(f"[*] T2IRewardPipeline.__call__() error: {e}")
            return (
                np.zeros((1, len(tokenized_prompt[0])), dtype=np.float32),
                np.zeros((1, len(tokenized_prompt[0])), dtype=np.float32),
                np.zeros((1, len(tokenized_prompt[0])), dtype=np.float32),
                np.zeros((1, len(tokenized_prompt[0])), dtype=np.float32),
                np.zeros((1, len(tokenized_prompt[0])), dtype=np.float32),
            )
