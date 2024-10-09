import base64
import io
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from albumentations.pytorch import ToTensorV2
from loguru import logger
from PIL import Image
from triton_python_backend_utils import (
    ModelConfig,
    get_input_tensor_by_name,
    get_output_config_by_name,
    triton_string_to_numpy,
)


def load_image_from_b64(image_b64: str) -> Image.Image:
    try:
        image_byte = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_byte)).convert("RGB")
    except:
        image = Image.new("RGB", (224, 224))
        logger.error("Error loading image")
    return image


def from_image_to_b64(img: Image.Image) -> bytes:
    image_arr = np.array(img)[..., ::-1]
    _, imgByteArr = cv2.imencode(".png", image_arr)
    encoded = base64.b64encode(imgByteArr)
    return encoded


image_size = 224
min_scale = 0.5
max_scale = 1.0
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
transform = A.Compose(
    [
        A.RandomResizedCrop(
            image_size,
            image_size,
            scale=(min_scale, max_scale),
            interpolation=cv2.INTER_CUBIC,
        ),
        A.HorizontalFlip(),
    ]
)
post_transform = A.Compose(
    [
        A.Normalize(mean, std),
        ToTensorV2(),
    ]
)


def preprocess(b64_image) -> Tuple[Image.Image, Image.Image, torch.Tensor]:
    image = load_image_from_b64(b64_image)
    crop_image = transform(image=np.array(image))["image"]
    tensor_image = post_transform(image=crop_image)["image"]
    return image, crop_image, tensor_image


class TritonPythonModel:
    def initialize(self, args) -> None:
        self.model_config = ModelConfig(args["model_config"])
        self.device = f"cuda:{args['model_instance_device_id']}" if args["model_instance_kind"] == "GPU" else "cpu"
        logger.info(f"{args['model_name']}(v{args['model_version']}, {self.device}) loaded successfully")

    def execute(self, requests) -> List:
        Tensor = pb_utils.Tensor
        InferenceResponse = pb_utils.InferenceResponse
        requests = [
            {
                "IMAGE": pb_utils.get_input_tensor_by_name(request, "IMAGE").as_numpy().squeeze(1),
            }
            for request in requests
        ]
        logger.debug(f"Received {[request['IMAGE'].shape[0] for request in requests]} images")
        responses = []
        for request in requests:
            image, crop_image, tensor_image = [*zip(*map(preprocess, request["IMAGE"]))]
            crop_image = np.array(crop_image)
            tensor_image = torch.stack(tensor_image, dim=0)
            crop_image = Tensor("CROP_IMAGE", crop_image)
            tensor_image = Tensor("TENSOR_IMAGE", tensor_image.numpy())
            responses.append(InferenceResponse(output_tensors=[crop_image, tensor_image]))
        return responses

    def finalize(self) -> None:
        logger.info("Cleaning up...")
