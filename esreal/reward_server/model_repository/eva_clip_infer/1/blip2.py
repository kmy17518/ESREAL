import torch
import torch.nn as nn
from clip_vit import create_clip_vit_L
from eva_vit import create_eva_vit_g

# from lavis.models import clip_vit, eva_vit

# create_eva_vit_g = eva_vit.create_eva_vit_g
# create_clip_vit_L = clip_vit.create_clip_vit_L


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def init_vision_encoder(
    model_name="eva_clip_g",
    img_size=224,
    drop_path_rate=0.0,
    use_grad_checkpoint=False,
    precision="fp16",
    device="cuda",
):
    assert model_name in [
        "eva_clip_g",
        "eva2_clip_L",
        "clip_L",
    ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
    if model_name == "eva_clip_g":
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision, device
        )
    #         elif model_name == "eva2_clip_L":
    #             visual_encoder = create_eva2_vit_L(
    #                 img_size, drop_path_rate, use_grad_checkpoint, precision
    #             )
    elif model_name == "clip_L":
        visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
    ln_vision = LayerNorm(visual_encoder.num_features)
    return visual_encoder, ln_vision
