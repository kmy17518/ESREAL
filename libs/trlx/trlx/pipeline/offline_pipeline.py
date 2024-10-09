import base64
import io
from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, Iterable, List, Tuple, Union

from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms.functional as VF
import tritonclient.http as httpclient
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from tritonclient.utils import np_to_triton_dtype

from trlx.data.ilql_types import (
    ILQLBatch,
    ILQLElement,
    ILQLSeq2SeqBatch,
    ILQLSeq2SeqElement,
)
from trlx.pipeline import BasePipeline, BaseRolloutStore, register_datapipeline


@dataclass
class DialogMessage:
    is_output: bool
    tokens: Tuple[int]


def tokenize_dialogue(  # noqa: C901
    dialogue: Union[str, Iterable[str]],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    max_length=2048,
) -> List[DialogMessage]:
    """
    Tokenize sample with the interleaved form of (prompt_1, output_1, prompt_2, output_2...)
    """
    if isinstance(dialogue, str):
        bos_token = tokenizer.bos_token or tokenizer.eos_token
        dialogue = [bos_token, dialogue]
    elif isinstance(dialogue, Iterable):
        if len(dialogue) % 2 != 0:
            raise ValueError("Dialogue must have an even number of phrases, alternating prompt and output")
        dialogue = list(dialogue)

    if not dialogue[-1].endswith(tokenizer.eos_token):
        dialogue[-1] = dialogue[-1] + tokenizer.eos_token

    tokenized = [
        DialogMessage(
            is_output=i % 2 == 1,
            tokens=tuple(tokenizer(dialogue[i], add_special_tokens=False).input_ids),
        )
        for i in range(len(dialogue))
    ]

    # flip to truncate from the left
    if tokenizer.truncation_side == "left":
        tokenized = [DialogMessage(is_output=m.is_output, tokens=m.tokens[::-1]) for m in tokenized[::-1]]

    # truncate if necessary
    lengths = [len(t.tokens) for t in tokenized]
    cumsum_lengths = [sum(lengths[:i]) for i in range(len(lengths))]
    truncated = [
        DialogMessage(is_output=t.is_output, tokens=t.tokens[: max(max_length - cl, 0)])
        for t, cl in zip(tokenized, cumsum_lengths)
    ]

    # flip back if was fliped to left truncate
    if tokenizer.truncation_side == "left":
        truncated = [DialogMessage(is_output=m.is_output, tokens=m.tokens[::-1]) for m in truncated[::-1]]

    # remove empty messages
    out = [t for t in truncated if len(t.tokens) > 0]

    if out[0].is_output:
        if sum(map(lambda msg: len(msg.tokens), out)) == max_length:
            if tokenizer.truncation_side == "left":
                out[0].tokens = out[0].tokens[1:]
            else:
                out[-1].tokens = out[-1].tokens[:-1]

        out.insert(0, DialogMessage(False, (tokenizer.bos_token_id,)))
    return out


class DialogStore(BaseRolloutStore):
    def __init__(self, dialogs: List[List[DialogMessage]], tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        attention_masks = [torch.ones(sum(len(m.tokens) for m in d), dtype=torch.bool) for d in dialogs]
        input_ids = [torch.tensor([t for m in d for t in m.tokens], dtype=torch.long) for d in dialogs]
        # -100 is the ignore index for CrossEntropyLoss
        labels = [
            torch.tensor(
                [t if m.is_output else -100 for m in d for t in m.tokens],
                dtype=torch.long,
            )
            for d in dialogs
        ]
        self.history = [
            dict(input_ids=i, attention_mask=a, labels=l) for i, a, l in zip(input_ids, attention_masks, labels)
        ]

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        hf_collate_fn = DataCollatorWithPadding(self.tokenizer)

        def collate_fn(elems: Iterable[dict]):
            batch = hf_collate_fn(
                {
                    "input_ids": [e["input_ids"] for e in elems],
                    "attention_mask": [e["attention_mask"] for e in elems],
                }
            )
            labels = hf_collate_fn([{"input_ids": e["labels"]} for e in elems])["input_ids"]
            batch["labels"] = labels
            return batch

        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)


@register_datapipeline
class PromptPipeline(BasePipeline):
    """
    Dataloader which is used to supply prompts for either training or evaluation

    Args:
        prompts (`List[str]` or `List[Dict[str, Any]]`): list of raw text prompts or a dictionary with a required
            key `"prompt"` and extra information, that would be passed along the generation for that prompt as a
            keyword argument to a reward function.
        max_prompt_length (`int`): max length of the prompt, if exceeded the prompt will be truncated according to
            tokenizer's truncation setting.
        tokenizer (`transformers.PreTrainedTokenizer`): a tokenizer to tokenize prompts with.
        add_special_tokens (`bool`): whether to encode prompts with tokenizer's special tokens (passed directly
            into `tokenizer.encode`)
    """

    def __init__(
        self,
        prompts: Union[List[Dict[str, Any]], List[str]],
        max_prompt_length: int,
        tokenizer: PreTrainedTokenizer,
        add_special_tokens: bool = False,
    ):
        super().__init__()

        if isinstance(prompts[0], dict):
            metadata = prompts
            prompts = [x.pop("prompt") for x in metadata]
        else:
            metadata = [{}] * len(prompts)

        model_inputs = tokenizer(
            prompts,
            truncation=True,
            padding=False,
            max_length=max_prompt_length,
            add_special_tokens=add_special_tokens,
        )

        prompts_tokens = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        self.tokenizer = tokenizer
        self.prompts = [
            {"input_ids": tokens, "attention_mask": mask, **metadata}
            for tokens, mask, metadata in zip(prompts_tokens, attention_mask, metadata)
        ]

    def __getitem__(self, ix: int):
        return self.prompts[ix]

    def __len__(self) -> int:
        return len(self.prompts)

    def create_loader(self, batch_size: int, shuffle=False, sampler=None, drop_last=False) -> DataLoader:
        def collate_fn(xs):
            out = self.tokenizer.pad([{"input_ids": x["input_ids"]} for x in xs], return_tensors="pt")

            for key in xs[0]:
                if key != "input_ids" and key != "attention_mask":
                    out[key] = [x[key] for x in xs]

            return out

        # Since all data is already pre-processed, no need to have
        # multi-process data loading
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            drop_last=drop_last,
        )


@register_datapipeline
class VLMDatasetPipeline(BasePipeline):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        max_prompt_length: int,
        tokenizer: PreTrainedTokenizer,
        add_special_tokens: bool = False,
        triton_server_url: str = "localhost:8000",
        args=None,
        eval=False,
    ):
        super().__init__()

        self.dataset = dataset
        self.max_prompt_length = max_prompt_length
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.triton_server_url = triton_server_url
        self.task_name = args.task_name
        self.eval = eval

    def from_image_to_b64(self, img):
        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format="PNG")
        imgByteArr = imgByteArr.getvalue()
        encoded = base64.b64encode(imgByteArr)
        return encoded

    def request_vit(self, images: List[Image.Image]):
        b64_images = np.array([[self.from_image_to_b64(image)] for image in images])

        with httpclient.InferenceServerClient(
            self.triton_server_url, connection_timeout=600, network_timeout=600
        ) as client:
            inputs = [
                httpclient.InferInput("IMAGE", b64_images.shape, np_to_triton_dtype(b64_images.dtype)),
            ]

            inputs[0].set_data_from_numpy(b64_images)

            outputs = [
                httpclient.InferRequestedOutput("CROP_IMAGE"),
                httpclient.InferRequestedOutput("IMAGE_EMBEDS"),
            ]

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = client.infer("eva_clip", inputs, request_id=str(1), outputs=outputs)
                    crop_image = response.as_numpy("CROP_IMAGE")  # (B, 224, 224, 3)
                    image_embeds = response.as_numpy("IMAGE_EMBEDS")  # (B, 257, 1408)
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Retrying due to error: {e}")
                        continue
                    else:
                        raise e

        return crop_image, image_embeds

    def __getitem__(self, ix: int):
        if self.task_name == "short_caption":
            instruction = "Write a short description for the image."
        elif self.task_name == "long_caption":
            instruction = "Write a detailed description for the image."
        elif self.task_name == "vqa" and "question" in self.datasetp[ix]:
            instruction = self.dataset[ix]["question"]
        else:
            raise ValueError(f"Task {self.task_name} not supported.")

        model_inputs = self.tokenizer(
            instruction,
            truncation=True,
            padding=False,
            max_length=self.max_prompt_length,
            add_special_tokens=self.add_special_tokens,
        )

        crop_image, image_embeds = self.request_vit([self.dataset[ix]["image"]])

        prompt = {
            "image": torch.from_numpy(crop_image).squeeze(0),
            "image_emb": torch.from_numpy(image_embeds).squeeze(0),
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
        }

        return prompt

    def __len__(self) -> int:
        return len(self.dataset)

    def create_loader(self, batch_size: int, shuffle=False, sampler=None, drop_last=False) -> DataLoader:
        def collate_fn(xs):
            out = self.tokenizer.pad([{"input_ids": x["input_ids"]} for x in xs], return_tensors="pt")

            out["image"] = torch.stack([x["image"] for x in xs])
            out["image_emb"] = torch.stack([x["image_emb"] for x in xs])

            for key in xs[0]:
                if key not in ["image", "image_emb", "input_ids", "attention_mask"]:
                    out[key] = [x[key] for x in xs]

            return out

        # Since all data is already pre-processed, no need to have
        # multi-process data loading
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=16 if not self.eval else 0,
            drop_last=drop_last,
        )


def ilql_collate_fn(elems: Iterable[ILQLElement]):
    return ILQLBatch(
        pad_sequence([x.input_ids for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.attention_mask for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.rewards for x in elems], batch_first=True, padding_value=0.0),
        pad_sequence([x.states_ixs for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.actions_ixs for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.dones for x in elems], batch_first=True, padding_value=0),
    )


class ILQLRolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training ILQL
    """

    def __init__(self, input_ids, attention_mask, rewards, states_ixs, actions_ixs, dones):
        super().__init__()

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.rewards = rewards
        self.states_ixs = states_ixs
        self.actions_ixs = actions_ixs
        self.dones = dones

    def __getitem__(self, ix: int) -> ILQLElement:
        return ILQLElement(
            self.input_ids[ix],
            self.attention_mask[ix],
            self.rewards[ix],
            self.states_ixs[ix],
            self.actions_ixs[ix],
            self.dones[ix],
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def create_loader(self, batch_size: int):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ilql_collate_fn,
            drop_last=torch.distributed.is_initialized(),
        )


def ilql_seq2seq_collate_fn(elems: Iterable[ILQLElement]):
    return ILQLSeq2SeqBatch(
        pad_sequence([x.input_ids for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.attention_mask for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.decoder_input_ids for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.rewards for x in elems], batch_first=True, padding_value=0.0),
        pad_sequence([x.states_ixs for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.actions_ixs for x in elems], batch_first=True, padding_value=0),
        pad_sequence([x.dones for x in elems], batch_first=True, padding_value=0),
    )


class ILQLSeq2SeqRolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training ILQL
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        rewards,
        states_ixs,
        actions_ixs,
        dones,
    ):
        super().__init__()

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.decoder_input_ids = decoder_input_ids
        self.rewards = rewards
        self.states_ixs = states_ixs
        self.actions_ixs = actions_ixs
        self.dones = dones

    def __getitem__(self, ix: int) -> ILQLElement:
        return ILQLSeq2SeqElement(
            self.input_ids[ix],
            self.attention_mask[ix],
            self.decoder_input_ids[ix],
            self.rewards[ix],
            self.states_ixs[ix],
            self.actions_ixs[ix],
            self.dones[ix],
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def create_loader(self, batch_size: int):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ilql_seq2seq_collate_fn,
            drop_last=torch.distributed.is_initialized(),
        )
