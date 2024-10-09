import io
import pickle
from typing import List, Optional, Tuple

import numpy as np
import open_clip
import torch
from PIL import Image

from .registry import registry

# def open_clip(images: np.ndarray) -> np.ndarray:
#     import triton_python_backend_utils as pb_utils

#     images = pb_utils.Tensor("images", images)
#     infer_request = pb_utils.InferenceRequest(
#         model_name="open_clip",
#         requested_output_names=["pooler_output"],
#         inputs=[images],
#     )
#     inference_response = infer_request.exec()
#     if inference_response.has_error():
#         raise pb_utils.TritonModelException(inference_response.error().message())
#     pooler_output = pb_utils.get_output_tensor_by_name(inference_response, "pooler_output").as_numpy()
#     return pooler_output


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


def open_clip(
    images: Optional[List[Image.Image]] = None, text: Optional[List[str]] = None, normalize=True
) -> Tuple[torch.Tensor, torch.Tensor]:
    import triton_python_backend_utils as pb_utils

    INPUT = encode({"images": images, "text": text})[None]
    infer_request = pb_utils.InferenceRequest(
        model_name="open_clip",
        requested_output_names=["FINAL_OUTPUT"],
        inputs=[pb_utils.Tensor("INPUT", INPUT)],
    )
    inference_response = infer_request.exec()
    if inference_response.has_error():
        raise pb_utils.TritonModelException(inference_response.error().message())
    output = pb_utils.get_output_tensor_by_name(inference_response, "FINAL_OUTPUT").as_numpy()
    output = decode(output)
    image_features, text_features, logit_scale_exp = (
        output["image_embeds"],
        output["text_embeds"],
        output["logit_scale_exp"],
    )
    if normalize:
        if image_features is not None:
            image_features /= np.linalg.norm(image_features, axis=-1, keepdims=True)
        if text_features is not None:
            text_features /= np.linalg.norm(text_features, axis=-1, keepdims=True)
    return image_features, text_features


@registry.register("clip")
class CLIP:
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "datacomp_xl_s13b_b90k"):
        try:
            import triton_python_backend_utils as pb_utils

            type(self).__call__ = lambda self, *args, **kwargs: open_clip(*args, **kwargs)
            return
        except ImportError:
            ...
        self.device = registry.device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, jit=False, device=self.device, precision="fp16"
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def __call__(
        self, images: Optional[List[Image.Image]] = None, text: Optional[List[str]] = None, normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(text, str):
            text = [text]
        images = torch.stack([*map(self.preprocess, images)]).to(self.device) if images else None
        text = self.tokenizer(text).to(self.device) if text else None
        with torch.cuda.amp.autocast():
            image_features, text_features, logit_scale_exp = self.model(images, text)
        if image_features is not None:
            if normalize:
                image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.detach().cpu().numpy()
        if text_features is not None:
            if normalize:
                text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.detach().cpu().numpy()
        return image_features, text_features


if __name__ == "__main__":
    from diffusers.utils import load_image
    from utils import time_measure

    image = load_image("https://images6.alphacoders.com/337/337780.jpg")
    text = ["a photo of a cat", "a photo of a dog"]
    for _ in range(5):
        image_features, text_features = clip(images=image, text=text)
    with time_measure("clip"):
        for _ in range(10):
            # 0.753 for jit=False, 0.641 for jit=True
            image_features, text_features = clip(images=image, text=text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print(similarity)
