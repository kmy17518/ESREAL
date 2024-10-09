import numpy as np
import open_clip
import torch
import triton_python_backend_utils as pb_utils
from loguru import logger
from triton_python_backend_utils import ModelConfig

from .utils import time_measure


class TritonPythonModel:
    def initialize(self, args) -> None:
        self.model_config = ModelConfig(args["model_config"])
        self.device = f"cuda:{args['model_instance_device_id']}" if args["model_instance_kind"] == "GPU" else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="datacomp_xl_s13b_b90k", jit=False, device=self.device, precision="fp16"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        logger.info(f"{args['model_name']}(v{args['model_version']}, {self.device}) loaded successfully")

    @time_measure("execute")
    def execute(self, requests):
        # fmt: off
        requests = [
            {
                "images": pb_utils.get_input_tensor_by_name(request, "images").as_numpy(),
                "text": pb_utils.get_input_tensor_by_name(request, "text").as_numpy().squeeze(-1),
            }
            for request in requests
        ]
        # fmt: on
        logger.debug(f"Received {[request['images'].shape[0] for request in requests]} images")
        responses = []
        batched_inputs = {"images": [], "text": []}
        batched_dims = []
        for request in requests:
            batched_inputs["images"].append(request["images"])
            batched_inputs["text"].append(request["text"])
            batched_dims.append(request["images"].shape[0])
        images = np.concatenate(batched_inputs["images"])
        text = np.concatenate(batched_inputs["text"])
        image_indices = np.where(text == b"")[0]
        text_indices = np.where(text != b"")[0]
        with torch.inference_mode(), torch.cuda.amp.autocast():
            _images = torch.from_numpy(images).to(self.device)
            _text = self.tokenizer([*map(bytes.decode, text[text_indices])]).to(self.device)
            _images = _images if _images.shape[0] > 0 else None
            _text = _text if _text.shape[0] > 0 else None
            with torch.no_grad(), torch.cuda.amp.autocast(), time_measure("inference"):
                # return image_features, text_features, self.logit_scale.exp()
                image_embeds, text_embeds, _ = self.model(_images, _text)
        # concat two result embedding by image_indices and text_indices
        pooler_output = np.zeros((images.shape[0], 768), dtype=np.float32)
        if image_embeds is not None:
            pooler_output[image_indices] = image_embeds.detach().cpu().numpy()
        if text_embeds is not None:
            pooler_output[text_indices] = text_embeds.detach().cpu().numpy()
        pooler_output = np.split(pooler_output, np.cumsum(batched_dims)[:-1])
        for pool in pooler_output:
            pool = pb_utils.Tensor("pooler_output", pool)
            responses.append(pb_utils.InferenceResponse(output_tensors=[pool]))
        return responses

    def finalize(self) -> None:
        logger.info("Cleaning up...")
