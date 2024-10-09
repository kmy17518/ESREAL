import os
from typing import List

from PIL import Image
from loguru import logger
import triton_python_backend_utils as pb_utils
from triton_python_backend_utils import ModelConfig


class TritonPythonModel:
    def initialize(self, args) -> None:
        self.model_config = ModelConfig(args["model_config"])
        orig_device_id = args["model_instance_device_id"]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(orig_device_id)
        self.device = "cuda:0"

        from .reward_model.registry import registry

        registry.device = self.device
        self.model = registry.reward_pipeline
        registry.initialize()
        
        logger.info(f"{args['model_name']}(v{args['model_version']}, cuda:{orig_device_id}) loaded successfully")

    def execute(self, requests) -> List:
        Tensor = pb_utils.Tensor
        InferenceResponse = pb_utils.InferenceResponse
        requests = [
            {
                "IMAGE": pb_utils.get_input_tensor_by_name(request, "IMAGE").as_numpy(),
                "PROMPT": pb_utils.get_input_tensor_by_name(request, "PROMPT").as_numpy().squeeze(-1),
                "TOKENIZED_PROMPT": pb_utils.get_input_tensor_by_name(request, "TOKENIZED_PROMPT").as_numpy(),
            }
            for request in requests
        ]
        responses = []
        for request in requests:
            images = [*map(Image.fromarray, request["IMAGE"])]
            prompts = [*map(bytes.decode, request["PROMPT"])]
            tokenized_prompt = [[*map(bytes.decode, tkpt)] for tkpt in request["TOKENIZED_PROMPT"]]
            (
                mean_rec_reward,
                mean_obj_penalty,
                mean_att_penalty,
                mean_rel_penalty,
                mean_pos_penalty
            ) = self.model(images, prompts, tokenized_prompt)
            responses.append(
                InferenceResponse(
                    output_tensors=[
                        Tensor("MEAN_REC_REWARD", mean_rec_reward),
                        Tensor("MEAN_OBJ_PENALTY", mean_obj_penalty),
                        Tensor("MEAN_ATT_PENALTY", mean_att_penalty),
                        Tensor("MEAN_REL_PENALTY", mean_rel_penalty),
                        Tensor("MEAN_POS_PENALTY", mean_pos_penalty),
                    ]
                )
            )
        return responses

    def finalize(self) -> None:
        logger.info("Cleaning up...")
