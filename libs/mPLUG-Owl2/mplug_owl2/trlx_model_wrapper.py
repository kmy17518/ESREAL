import os

import torch
from torch import nn

from trlx.models.modeling_ppo import (
    CausalLMOutputWithValue,
)

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from mplug_owl2.model import *
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX


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


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class MPLUGOwl2LlamaForCausalLMWithValueHead(MPLUGOwl2LlamaForCausalLM):
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
            if 'vision_abstractor' not in name: ### needs revision 
                param.requires_grad = False
        self.model.vision_model = self.model.vision_model.eval()
        self.model.vision_model.train = disabled_train

    def forward_trlx( # from accelerate_vlm_ppo_trainer
            self,
            input_ids=None,
            prompt_input_ids=None,
            attention_mask=None,
            position_ids=None,
            return_dict=True,
            images=None,
            get_ref_logits=False,
            **kwargs
    ):
        output_hidden_states = kwargs.pop('output_hidden_states', True)
        output_hidden_states = True # overwrite

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            labels=None,
            return_dict=return_dict,
            output_hidden_states=output_hidden_states,
            get_ref_logits=get_ref_logits,
            **kwargs
        )
        
        start_idx = prompt_input_ids.shape[1] + 65 - 1 # llava: 576

        logits = outputs.logits[:, start_idx:, :]
        value = self.value_head(outputs.hidden_states[-3][:, start_idx:, :]).squeeze(-1)

        if return_dict:
            outputs_with_value = CausalLMOutputWithValue(
                logits=logits,
                value=value,
            )
            return outputs_with_value

        return logits, value

    def save_pretrained(self, directory=None, **kwargs):
        if directory is None:
            raise ValueError("A directory must be provided to save the model.")

        os.makedirs(directory, exist_ok=True)
        checkpoint_file = os.path.join(directory, "model_checkpoint.pth")

        # Save model's state dictionary
        torch.save(self.state_dict(), checkpoint_file)
        print(f"Model checkpoint saved at: {checkpoint_file}")
