import os

import torch
from torch import nn

from trlx.models.modeling_ppo import (
    CausalLMOutputWithValue,
)

from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

        self.value_head = nn.Linear(self.model.config.hidden_size, 1)

    def forward(
        self,
        image_emb=None,
        prompt_input_ids=None,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        return_dict=True,
    ):
        outputs = self.model(input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
        logits = outputs.logits
        value = self.value_head(outputs.hidden_states[-3]).squeeze(-1)

        if return_dict:
            outputs_with_value = CausalLMOutputWithValue(
                logits=logits,
                value=value,
            )
            return outputs_with_value

        return logits, value
    
    def generate(self, image_emb, input_ids, attention_mask=None, **kwargs):
        prompt = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # samples = {
        #     "image": image_emb,
        #     "prompt": prompt,
        # }

        # if "do_sample" in kwargs:
        #     kwargs["use_nucleus_sampling"] = kwargs["do_sample"]
        #     del kwargs["do_sample"]

        return self.model.generate(input_ids, **kwargs)
    
    def save_pretrained(self, directory=None, **kwargs):
        self.model.save_pretrained(save_directory=directory, **kwargs)
    #     if directory is None:
    #         raise ValueError("A directory must be provided to save the model.")

    #     os.makedirs(directory, exist_ok=True)
    #     checkpoint_file = os.path.join(directory, "model_checkpoint.pth")

    #     # Save model's state dictionary
    #     torch.save(self.state_dict(), checkpoint_file)
    #     print(f"Model checkpoint saved at: {checkpoint_file}")

    
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class LlavaLlamaForCausalLMWithValueHead(LlavaLlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        hidden_dim = getattr(config, "hidden_size")
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 1, dtype=torch.bfloat16),
        )
        self.value_head = nn.Linear(self.model.config.hidden_size, 1)

        for name, param in self.model.named_parameters():
            if 'mm_projector' not in name:
                param.requires_grad = False
        self.model.vision_tower = self.model.vision_tower.eval()
        self.model.vision_tower.train = disabled_train

    def forward_trlx( # from accelerate_vlm_ppo_trainer
            self,
            image_emb=None,
            prompt_input_ids=None,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            return_dict=True,
            get_ref_logits=False,
            **kwargs
    ):
        inputs_embeds = kwargs.pop('inputs_embeds', None)
        output_hidden_states = kwargs.pop('output_hidden_states', True)
        output_hidden_states = True # overwrite

        # CHECK: labels are shifted?
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal_with_image_emb(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                image_emb=image_emb,
                get_ref_logits=get_ref_logits
            )
        else:
            labels = None

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=return_dict,
            output_hidden_states=output_hidden_states,
            **kwargs
        )

        start_idx = prompt_input_ids.shape[1] + 576 - 1

        logits = outputs.logits[:, start_idx:, :]
        value = self.value_head(outputs.hidden_states[-3][:, start_idx:, :]).squeeze(-1)

        if return_dict:
            outputs_with_value = CausalLMOutputWithValue(
                logits=logits,
                value=value,
            )
            return outputs_with_value

        return logits, value
    
    def generate_trlx(self, image_emb, input_ids, attention_mask=None, **kwargs):
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels
        ) = self.prepare_inputs_labels_for_multimodal_with_image_emb(
            input_ids=input_ids,
            position_ids=kwargs.pop("position_ids", None),
            attention_mask=attention_mask,
            past_key_values=None,
            labels=None,
            image_emb=image_emb,
        )
        
        return super().generate(
            inputs=input_ids,
            images=None,
            image_sizes=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
    def save_pretrained(self, directory=None, **kwargs):
        if directory is None:
            raise ValueError("A directory must be provided to save the model.")

        os.makedirs(directory, exist_ok=True)
        checkpoint_file = os.path.join(directory, "model_checkpoint.pth")

        # Save model's state dictionary
        torch.save(self.state_dict(), checkpoint_file)
        print(f"Model checkpoint saved at: {checkpoint_file}")

    def prepare_inputs_labels_for_multimodal_with_image_emb( # adapted from llava
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        image_emb, get_ref_logits=False
    ):
        # vision_tower = self.get_vision_tower()
        # if vision_tower is None or images is None or input_ids.shape[1] == 1:
        #     return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # if type(images) is list or images.ndim == 5:
        #     if type(images) is list:
        #         images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
        #     concat_images = torch.cat([image for image in images], dim=0)
        #     image_features = self.encode_images(concat_images)
        #     split_sizes = [image.shape[0] for image in images]
        #     image_features = torch.split(image_features, split_sizes, dim=0)
        #     mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
        #     image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
        #     if mm_patch_merge_type == 'flat':
        #         image_features = [x.flatten(0, 1) for x in image_features]
        #     elif mm_patch_merge_type.startswith('spatial'):
        #         new_image_features = []
        #         for image_idx, image_feature in enumerate(image_features):
        #             if image_feature.shape[0] > 1:
        #                 base_image_feature = image_feature[0]
        #                 image_feature = image_feature[1:]
        #                 height = width = self.get_vision_tower().num_patches_per_side
        #                 assert height * width == base_image_feature.shape[0]
        #                 if image_aspect_ratio == 'anyres':
        #                     num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
        #                     image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
        #                 else:
        #                     raise NotImplementedError
        #                 if 'unpad' in mm_patch_merge_type:
        #                     image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        #                     image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        #                     image_feature = unpad_image(image_feature, image_sizes[image_idx])
        #                     image_feature = torch.cat((
        #                         image_feature,
        #                         self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
        #                     ), dim=-1)
        #                     image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        #                 else:
        #                     image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
        #                     image_feature = image_feature.flatten(0, 3)
        #                 image_feature = torch.cat((base_image_feature, image_feature), dim=0)
        #             else:
        #                 image_feature = image_feature[0]
        #                 if 'unpad' in mm_patch_merge_type:
        #                     image_feature = torch.cat((
        #                         image_feature,
        #                         self.model.image_newline[None].to(image_feature.device)
        #                     ), dim=0)
        #             new_image_features.append(image_feature)
        #         image_features = new_image_features
        #     else:
        #         raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        # else:
        #     image_features = self.encode_images(images)
        if get_ref_logits:
            image_features = self.get_model().frozen_mm_projector(image_emb)
        else:
            image_features = self.get_model().mm_projector(image_emb)
        
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        len_input_ids = input_ids.shape[1]
        # input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        # labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds) #### start debug
        # max_len = max(max_len, len_input_ids) #### debug
            
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
