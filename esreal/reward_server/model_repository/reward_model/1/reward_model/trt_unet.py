import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack


def unet(
    sample,
    timestep,
    encoder_hidden_states,
    timestep_cond=None,
    cross_attention_kwargs=None,
    added_cond_kwargs=None,
    return_dict=None,
):
    """
    in SDPipeline:
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
    in SDXLPipeline:
                # 0.21.4
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                # 0.23.0
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
    """
    text_embeds = added_cond_kwargs["text_embeds"]
    time_ids = added_cond_kwargs["time_ids"]
    device, dtype = sample.device, sample.dtype
    # dtype patching for TensorRT
    target_dtype = torch.float16
    sample = sample.to(dtype=target_dtype)
    timestep = timestep.to(dtype=target_dtype)
    encoder_hidden_states = encoder_hidden_states.to(dtype=target_dtype)
    text_embeds = text_embeds.to(dtype=target_dtype)
    time_ids = time_ids.to(dtype=target_dtype)

    timestep = timestep.detach().cpu().numpy().repeat(sample.shape[0]).reshape(-1, 1)
    sample = pb_utils.Tensor.from_dlpack("sample", to_dlpack(sample))  # [-1, 4, 96, 96]
    timestep = pb_utils.Tensor("timestep", timestep)  # [-1, 1]
    encoder_hidden_states = pb_utils.Tensor.from_dlpack(
        "encoder_hidden_states", to_dlpack(encoder_hidden_states)
    )  # [-1, 77, 2048]
    text_embeds = pb_utils.Tensor.from_dlpack("text_embeds", to_dlpack(text_embeds))  # [-1, 1280]
    time_ids = pb_utils.Tensor.from_dlpack("time_ids", to_dlpack(time_ids))  # [-1, 6]
    unet_request = pb_utils.InferenceRequest(
        model_name="unet",
        requested_output_names=["out_sample"],
        inputs=[sample, timestep, encoder_hidden_states, text_embeds, time_ids],
    )

    response = unet_request.exec()
    if response.has_error():
        raise pb_utils.TritonModelException(response.error().message())
    else:
        out_sample = pb_utils.get_output_tensor_by_name(response, "out_sample")
    out_sample = from_dlpack(out_sample.to_dlpack()).clone()
    out_sample = out_sample.to(device=device, dtype=dtype)
    return [out_sample]


# make unet.add_embedding.linear_1.in_features to return 2816
unet.add_embedding = type("add_embedding", (), {"linear_1": type("linear_1", (), {"in_features": 2816})})
