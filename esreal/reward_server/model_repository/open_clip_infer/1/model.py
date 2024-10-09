import io
import pickle

import numpy as np
import open_clip
import torch
import triton_python_backend_utils as pb_utils
from loguru import logger
from triton_python_backend_utils import ModelConfig


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
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="datacomp_xl_s13b_b90k", jit=False, device=self.device, precision="fp16"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        # jit takes 8, 16, 15, 16, 0, 0, 0, 0 seconds for 1~8 images
        compile = False
        if compile:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            images = torch.randn(1, 3, 224, 224).to(self.device)
            texts = torch.randint(1, 100, size=(1, 77)).to(self.device)
            with torch.inference_mode(), torch.cuda.amp.autocast():
                for idx in range(10):
                    self.model(images, texts)
                    logger.info(f"Compiled model {idx}")
        logger.info(f"{args['model_name']}(v{args['model_version']}, {self.device}) loaded successfully")

    def execute(self, requests):
        logger.info(f"Received {len(requests)} requests")
        # Concatenate all images and texts from the requests
        all_images = []
        all_images_len = []
        all_texts = []
        all_texts_len = []
        shapes = []
        maybe_len = lambda x: x.shape[0] if x is not None else 0
        for request in requests:
            inputs = decode(pb_utils.get_input_tensor_by_name(request, "INPUT").as_numpy())
            shapes.append(pb_utils.get_input_tensor_by_name(request, "INPUT").as_numpy().shape)
            images, text = inputs["images"], inputs["text"]
            if images is not None:
                all_images.append(images)
            if text is not None:
                all_texts.append(text)
            all_images_len.append(maybe_len(images))
            all_texts_len.append(maybe_len(text))
        logger.debug(shapes)
        # Process the concatenated images and texts
        all_images = torch.cat(all_images).to(self.device) if len(all_images) > 0 else None
        all_texts = torch.cat(all_texts).to(self.device) if len(all_texts) > 0 else None
        assert maybe_len(all_images) + maybe_len(all_texts) < 1024, "Batch size should be less than 1024"

        with torch.inference_mode(), torch.cuda.amp.autocast():
            # return image_features, text_features, self.logit_scale.exp()
            all_image_embeds, all_text_embeds, logit_scale_exp = self.model(all_images, all_texts)
            all_image_embeds = all_image_embeds.detach().cpu().numpy() if all_image_embeds is not None else None
            all_text_embeds = all_text_embeds.detach().cpu().numpy() if all_text_embeds is not None else None
            logit_scale_exp = logit_scale_exp.detach().cpu().numpy()

        responses = []
        images_len_sum = texts_len_sum = 0
        for images_len, texts_len in zip(all_images_len, all_texts_len):
            if all_image_embeds is not None and images_len > 0:
                image_embeds = all_image_embeds[images_len_sum : images_len_sum + images_len]
            else:
                image_embeds = None

            if all_text_embeds is not None and texts_len > 0:
                text_embeds = all_text_embeds[texts_len_sum : texts_len_sum + texts_len]
            else:
                text_embeds = None

            output = encode(
                {"image_embeds": image_embeds, "text_embeds": text_embeds, "logit_scale_exp": logit_scale_exp}
            )
            output = pb_utils.Tensor("OUTPUT", output)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output]))
            images_len_sum += images_len
            texts_len_sum += texts_len
        return responses

    def finalize(self) -> None:
        logger.info("Cleaning up...")
