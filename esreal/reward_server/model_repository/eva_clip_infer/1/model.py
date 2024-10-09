import base64
import io
import multiprocessing as mp
from typing import Callable, List, Tuple

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from loguru import logger
from PIL import Image
from triton_python_backend_utils import (
    ModelConfig,
)
from utils import time_measure


def load_model(device: torch.device) -> Callable:
    from blip2 import init_vision_encoder

    visual_encoder, ln_vision = init_vision_encoder()
    visual_encoder.to(device)
    ln_vision.to(device)
    visual_encoder.eval()
    ln_vision.eval()

    if torch.device(device).type == "cpu":
        # convert visual_encoer and ln_vision into fp32
        logger.info("[*] device=cpu; converting visual_encoder and ln_vision into fp32")
        visual_encoder = visual_encoder.float()
        ln_vision = ln_vision.float()

    def f(batch):
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                return ln_vision(visual_encoder(batch))

    return f


class TritonPythonModel:
    def initialize(self, args) -> None:
        self.model_config = ModelConfig(args["model_config"])
        self.device = f"cuda:{args['model_instance_device_id']}" if args["model_instance_kind"] == "GPU" else "cpu"
        self.model = load_model(self.device)
        logger.info(f"{args['model_name']}(v{args['model_version']}, {self.device}) loaded successfully")

    @time_measure("execute")
    def execute(self, requests) -> List:
        Tensor = pb_utils.Tensor
        InferenceResponse = pb_utils.InferenceResponse
        # fmt: off
        requests = [
            {
                "TENSOR_IMAGE": pb_utils.get_input_tensor_by_name(request, "TENSOR_IMAGE").as_numpy(),
            }
            for request in requests
        ]
        # fmt: on
        logger.debug(f"Received {[request['TENSOR_IMAGE'].shape[0] for request in requests]} images")
        responses = []
        batched_inputs = []
        batched_dims = []
        for request in requests:
            tensor_image = torch.from_numpy(request["TENSOR_IMAGE"])
            batched_inputs.append(tensor_image)
            batched_dims.append(tensor_image.shape[0])
        batched_inputs = torch.cat(batched_inputs, dim=0).to(self.device)
        image_embeds = self.model(batched_inputs).detach().cpu().numpy()
        image_embeds_split = np.split(image_embeds, np.cumsum(batched_dims)[:-1])
        for image_embeds in image_embeds_split:
            image_embeds = Tensor("IMAGE_EMBEDS", image_embeds)
            responses.append(InferenceResponse(output_tensors=[image_embeds]))
        return responses

    def finalize(self) -> None:
        logger.info("Cleaning up...")
