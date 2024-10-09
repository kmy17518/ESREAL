import torch
import numpy as np
from PIL import Image
from torchvision.ops import box_convert

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict


gdino_image_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def create_gdino(
    config_path: str = "esreal/reward_server/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    checkpoint_path: str = "esreal/reward_server/GroundingDINO/weights/groundingdino_swint_ogc.pth",
    device: str = "cuda:0",
):
    gdino_model = load_model(
        config_path,
        checkpoint_path,
        device=device,
    )
    return gdino_model


def convert_to_char_start_end_indices(
    boxes,
    phrases,
    logits,
    token_indices,
    tokenized,
):
    new_boxes = []
    new_phrases = []
    new_logits = []
    new_token_indices = []
    char_start_end_indices = []

    for idx, indices in enumerate(token_indices):
        if len(indices) == 0:
            continue

        new_boxes.append(boxes[idx])
        new_phrases.append(phrases[idx])
        new_logits.append(logits[idx])
        new_token_indices.append(indices)

        start_idx = tokenized.token_to_chars(indices[0])[0]
        end_idx = tokenized.token_to_chars(indices[-1])[1]
        char_start_end_indices.append([start_idx, end_idx])

    return new_boxes, new_phrases, new_logits, new_token_indices, char_start_end_indices


def run_gdino(
    image: Image.Image,
    prompt: str,
    gdino_model,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda:0",
):
    image = np.asarray(image)
    image = image.copy()  # Important!!!
    h, w, _ = image.shape

    transformed_image, _ = gdino_image_transform(image, None)
    transformed_image = transformed_image.to(device)

    boxes, logits, phrases, tokenized, token_indices = predict(
        model=gdino_model,
        image=transformed_image,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
        remove_combined=True,
    )
    boxes = boxes * torch.Tensor([w, h, w, h])
    boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")

    sorted_indices = sorted(range(len(logits)), key=lambda k: logits[k], reverse=True)
    boxes = [boxes[i].tolist() for i in sorted_indices]
    phrases = [phrases[i] for i in sorted_indices]
    logits = [logits[i].item() for i in sorted_indices]
    token_indices = [token_indices[i] for i in sorted_indices]

    boxes, phrases, logits, token_indices, char_start_end_indices = convert_to_char_start_end_indices(
        boxes,
        phrases,
        logits,
        token_indices,
        tokenized,
    )

    return boxes, phrases, char_start_end_indices
