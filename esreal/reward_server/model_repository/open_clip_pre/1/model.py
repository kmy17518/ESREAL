import io
import pickle

import albumentations as A
import cv2
import numpy as np
import open_clip
import torch
import triton_python_backend_utils as pb_utils
from albumentations.pytorch import ToTensorV2
from loguru import logger
from PIL import Image
from triton_python_backend_utils import ModelConfig


def from_npy_to_npy(image_npy: np.ndarray) -> np.ndarray:
    img_width = image_npy[0] | (image_npy[1] << 8)
    img_height = image_npy[2] | (image_npy[3] << 8)
    image_arr = image_npy[4:].reshape((img_height, img_width, -1))
    return image_arr


image_size = 224
min_scale = 0.5
max_scale = 1.0
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
"""
    Resize(size=224, interpolation=bicubic, max_size=None, antialias=warn)
    CenterCrop(size=(224, 224))
"""
transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=image_size, interpolation=cv2.INTER_CUBIC),
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean, std),
        ToTensorV2(),
    ]
)


def encode(inputs):
    buffer = io.BytesIO()
    pickle.dump(inputs, buffer)
    payload = buffer.getvalue()
    payload = np.frombuffer(payload, dtype=np.uint8)
    return payload


def decode(payload):
    payload = payload.tobytes()
    buffer = io.BytesIO(payload)
    inputs = pickle.load(buffer)
    return inputs


class TritonPythonModel:
    def initialize(self, args) -> None:
        self.model_config = ModelConfig(args["model_config"])
        self.device = f"cuda:{args['model_instance_device_id']}" if args["model_instance_kind"] == "GPU" else "cpu"
        # self.preprocess = open_clip.transform.image_transform(224, False, None, None)
        # _, _, self.preprocess = open_clip.create_model_and_transforms(
        #     "ViT-L-14", pretrained="datacomp_xl_s13b_b90k", jit=False, device=self.device, precision="fp16"
        # )
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        logger.info(f"{args['model_name']}(v{args['model_version']}, {self.device}) loaded successfully")

    def _execute(self, request):
        try:
            inputs = decode(pb_utils.get_input_tensor_by_name(request, "INPUT").as_numpy())
            images, text = inputs["images"], inputs["text"]
        except:
            nums = np.random.randint(1, 2)
            images = [Image.new("RGB", (224, 224), (0, 0, 0))] * nums
            text = ["a blue bird"] * nums
            logger.error("Failed to decode inputs, using test inputs")
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(text, str):
            text = [text]
        images = (
            torch.stack([transform(image=np.asarray(image))["image"] for image in images])
            if images is not None
            else None
        )
        text = self.tokenizer(text).to(self.device) if text is not None else None
        return encode({"images": images, "text": text})

    def execute(self, requests):
        logger.info(f"Received {len(requests)} requests")
        return [
            pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("OUTPUT", self._execute(request)[None])])
            for request in requests
        ]

    def finalize(self) -> None:
        logger.info("Cleaning up...")
