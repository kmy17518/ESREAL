import open_clip
import torch


class CLIPModel(torch.nn.Module):
    def __init__(self, model_name: str, pretrained: str = None):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        self.visual_model = CLIPVisualEncoder(model.visual)


class CLIPVisualEncoder(torch.nn.Module):
    # "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    def __init__(self, visual):
        super().__init__()
        self.model = visual
        self.attn_pool = self.model.attn_pool
        self.proj = self.model.proj
        self.model.output_tokens = True
        self.model.attn_pool = self.model.proj = None

    @torch.inference_mode()
    def forward(self, x):
        pooler_output, last_hidden_state = self.model(x)
        last_hidden_state = torch.cat([pooler_output.unsqueeze(1), last_hidden_state], dim=1)
        x = last_hidden_state
        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.model.ln_post(x)
            pooled, tokens = self.model._global_pool(x)
        else:
            pooled, tokens = self.model._global_pool(x)
            pooled = self.model.ln_post(pooled)

        if self.proj is not None:
            pooled = pooled @ self.proj
        return pooled, last_hidden_state


if __name__ == "__main__":
    # ViT-SO400M-14-SigLIP, webli
    # ViT-SO400M-14-SigLIP-384, webli
    model = None
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32, device="cuda")
    model.to("cuda")

    pooler_output, last_hidden_state = model(dummy_input)
    print(pooler_output.shape, last_hidden_state.shape)
