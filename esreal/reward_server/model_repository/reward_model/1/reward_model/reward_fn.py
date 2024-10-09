from typing import List
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .gdino_registry import run_gdino
from .registry import registry


###############
### Classes ###
###############

@dataclass
class Box:
    bbox: List[float]
    tokens: List[int]
    has_overlap: bool = False


@dataclass
class MergedBox:
    bbox_orig: List[float]
    bbox_recon: List[float]
    tokens: List[int]


#########################
### Utility Functions ###
#########################

def get_cos_sim(i1, i2):
    image_embeds, _ = registry.clip(images=[i1, i2])
    i2i_sim = image_embeds[0] @ image_embeds[1].T
    return i2i_sim
    

def get_char_token_map(tokenized_prompt, prompt):
    char_token_map = -np.ones((len(prompt)), dtype=np.int32)
    orig_str = prompt

    start_idx = 0
    for i, token in enumerate(tokenized_prompt):
        if len(token) == 0:
            continue
        
        while len(token) > 0:
            index = orig_str.find(token)
            if index != -1:
                char_token_map[start_idx+index:start_idx + index + len(token)] = i
                orig_str = orig_str[index+len(token):]
                start_idx += index + len(token)
                break
            else:
                token = token[1:]

        if len(token) == 0:
            raise ValueError(f"Tokenized prompt: {tokenized_prompt}, prompt: {prompt}")

    return char_token_map


def get_char_phrase_map(phrases, phrase_sentence, separator=" . "):
    char_phrase_map = -np.ones((len(phrase_sentence)), dtype=np.int32)
    start_idx = 0
    for i, phrase in enumerate(phrases):
        end_idx = start_idx + len(phrase) + len(separator)
        char_phrase_map[start_idx:end_idx] = i
        start_idx = end_idx
    return char_phrase_map


def get_token_range(start_index, end_index, char_token_map):
    sliced = char_token_map[start_index:end_index]  # glip: inclusive, gdino: exclusive
    unique = np.unique(sliced)
    unique = unique[unique != -1].tolist()
    return unique


def merge_boxes(box_list):
    x_left = min([box[0] for box in box_list])
    y_top = min([box[1] for box in box_list])
    x_right = max([box[2] for box in box_list])
    y_bottom = max([box[3] for box in box_list])
    return [x_left, y_top, x_right, y_bottom]


def has_overlap(box1, box2):
    return True if set(box1.tokens) & set(box2.tokens) else False


def merge_duplicate_phrases(
    bboxes,
    phrases,
    char_start_end_indices,
):
    item_dict = {}

    for i, (start_index, end_index) in enumerate(char_start_end_indices):
        if (start_index, end_index) not in item_dict:
            item_dict[(start_index, end_index)] = (bboxes[i], phrases[i])
        else:
            box_list = [bboxes[i], item_dict[(start_index, end_index)][0]]
            x_left = min([box[0] for box in box_list])
            y_top = min([box[1] for box in box_list])
            x_right = max([box[2] for box in box_list])
            y_bottom = max([box[3] for box in box_list])
            merged_bbox = [x_left, y_top, x_right, y_bottom]
            item_dict[(start_index, end_index)] = (merged_bbox, phrases[i])

    merged_bboxes = [item[0] for item in item_dict.values()]
    merged_phrases = [item[1] for item in item_dict.values()]
    merged_char_start_end_indices = list(item_dict.keys())
        
    return merged_bboxes, merged_phrases, merged_char_start_end_indices


def find_dot_indice(target_number, dot_indices):
        for dot_indice in dot_indices:
            if dot_indice >= target_number:
                return dot_indice
        return -1


def filter_the_image(bboxes, phrases, char_start_end_indices):
    filtered_bboxes = []
    filtered_phrases = []
    filtered_char_start_end_indices = []
    for i, phrase in enumerate(phrases):
        if phrase == "The image":
            continue
        filtered_bboxes.append(bboxes[i])
        filtered_phrases.append(phrases[i])
        filtered_char_start_end_indices.append(char_start_end_indices[i])
    return filtered_bboxes, filtered_phrases, filtered_char_start_end_indices


POS_TOKENS = [
    "left",
    "right",
    "top",
    "bottom",
    "center",
    "middle",
    "above",
    "below",
    "inside",
    "outside",
    "front",
    "behind",
    "upward",
    "downward",
    "up",
    "down",
    "inward",
    "outward",
    "over",
    "under",
]


#######################
### Reward Functions ##
#######################

class RewardCalculator:
    def __init__(
            self,
            gdino_model,
            device="cuda:0"
        ):
        self.gdino_model = gdino_model
        self.device = device

    def get_scores(self, sentence, tokenized_sentence, image, model_image):
        char_token_map = get_char_token_map(tokenized_sentence, sentence)

        bboxes_recon, phrases_recon, char_start_end_indices_recon = run_gdino(
            model_image,
            sentence,
            gdino_model=self.gdino_model,
            box_threshold=0.3,
            text_threshold=0.25,
            device=self.device,
        )

        bboxes_recon, phrases_recon, char_start_end_indices_recon = merge_duplicate_phrases(
            bboxes_recon,
            phrases_recon,
            char_start_end_indices_recon,
        )

        bboxes_recon, phrases_recon, char_start_end_indices_recon = filter_the_image(
            bboxes_recon,
            phrases_recon,
            char_start_end_indices_recon,
        )

        matched_phrases_recon = list(map(lambda x: get_token_range(x[0], x[1], char_token_map), char_start_end_indices_recon))
        box_list_recon = [Box(bbox, tokens, False) for bbox, tokens in zip(bboxes_recon, matched_phrases_recon)]

        separator = " . "
        end_separator = " ."
        phrases_recon_sentence = separator.join(phrases_recon) + end_separator
        char_phrase_map = get_char_phrase_map(phrases_recon, phrases_recon_sentence, separator=separator)
        
        bboxes_orig, phrases_orig, char_start_end_indices_orig = run_gdino(
            image,
            phrases_recon_sentence,
            self.gdino_model,
            box_threshold=0.2,
            text_threshold=0.01,
            device=self.device,
        )

        bboxes_orig, phrases_orig, char_start_end_indices_orig = merge_duplicate_phrases(
            bboxes_orig,
            phrases_orig,
            char_start_end_indices_orig,
        )

        merged_box_list = []
        matched_indices_for_recon = []

        for i, (start_index, end_index) in enumerate(char_start_end_indices_orig):
            sliced = char_phrase_map[start_index:end_index]
            unique = np.unique(sliced)
            unique = unique[unique != -1].tolist()

            if len(unique) == 0:
                continue

            matched_phrase_index = unique[-1]
            matched_indices_for_recon.append(matched_phrase_index)

            box1 = box_list_recon[matched_phrase_index]
            box2 = bboxes_orig[i]
            merged_box = MergedBox(bbox_recon=box1.bbox, bbox_orig=box2, tokens=box1.tokens)
            merged_box_list.append(merged_box)

        obj_penalty = np.zeros(len(tokenized_sentence))
        att_penalty = np.zeros(len(tokenized_sentence))
        rel_penalty = np.zeros(len(tokenized_sentence))
        pos_penalty = np.zeros(len(tokenized_sentence))

        # obj penalty
        unmatched_indices_for_recon = list(set(range(len(box_list_recon))) - set(matched_indices_for_recon))
        for unmatched_box_recon in [box_list_recon[i] for i in unmatched_indices_for_recon]:
            last_token_index = unmatched_box_recon.tokens[-1]
            obj_penalty[last_token_index] = -1

        # att penalty
        for merged_box in merged_box_list:
            last_token_index = merged_box.tokens[-1]
            cropped_image = image.crop(merged_box.bbox_orig)
            cropped_model_image = model_image.crop(merged_box.bbox_recon)
            att_penalty[last_token_index] = (get_cos_sim(cropped_image, cropped_model_image) - 1) / 2

        dot_indices = [i for i, token in enumerate(tokenized_sentence) if token.strip() == '.']

        # rel penalty
        # same_sentence_dict = {}

        # for i in range(len(merged_box_list)):
        #     box = merged_box_list[i]
        #     dot_indice = find_dot_indice(box.tokens[-1], dot_indices)
        #     if dot_indice not in same_sentence_dict:
        #         same_sentence_dict[dot_indice] = [box]
        #     else:
        #         same_sentence_dict[dot_indice].append(box)

        # for dot_indice, box_list in same_sentence_dict.items():
        #     if len(box_list) == 1:
        #         continue
        #     box1 = box_list[0]
        #     merged_bbox_orig = box1.bbox_orig
        #     merged_bbox_recon = box1.bbox_recon
        #     for box2 in box_list[1:]:
        #         merged_bbox_orig = merge_boxes([merged_bbox_orig, box2.bbox_orig])
        #         merged_bbox_recon = merge_boxes([merged_bbox_recon, box2.bbox_recon])
        #     cropped_image = image.crop(merged_bbox_orig)
        #     cropped_model_image = model_image.crop(merged_bbox_recon)
        #     rel_penalty[dot_indice] = (get_cos_sim(cropped_image, cropped_model_image) - 1) / 2

        # pos penalty
        same_sentence_dict = {}

        for i in range(len(merged_box_list)):
            box = merged_box_list[i]
            dot_indice = find_dot_indice(box.tokens[-1], dot_indices)
            if dot_indice not in same_sentence_dict:
                same_sentence_dict[dot_indice] = [box]
            else:
                same_sentence_dict[dot_indice].append(box)

        for dot_indice, box_list in same_sentence_dict.items():
            if len(box_list) == 2:
                for pos_token in POS_TOKENS:
                    if pos_token in tokenized_sentence:
                        pos_token_index = tokenized_sentence.index(pos_token)
                        if find_dot_indice(pos_token_index, dot_indices) == dot_indice:
                            box1 = box_list[0]
                            box2 = box_list[1]
                            box1_orig_center = [(box1.bbox_orig[0] + box1.bbox_orig[2]) / 2, (box1.bbox_orig[1] + box1.bbox_orig[3]) / 2]
                            box2_orig_center = [(box2.bbox_orig[0] + box2.bbox_orig[2]) / 2, (box2.bbox_orig[1] + box2.bbox_orig[3]) / 2]
                            box1_recon_center = [(box1.bbox_recon[0] + box1.bbox_recon[2]) / 2, (box1.bbox_recon[1] + box1.bbox_recon[3]) / 2]
                            box2_recon_center = [(box2.bbox_recon[0] + box2.bbox_recon[2]) / 2, (box2.bbox_recon[1] + box2.bbox_recon[3]) / 2]
                            orig_vector = [box2_orig_center[0] - box1_orig_center[0], box2_orig_center[1] - box1_orig_center[1]]
                            recon_vector = [box2_recon_center[0] - box1_recon_center[0], box2_recon_center[1] - box1_recon_center[1]]
                            cos_sim = (np.dot(orig_vector, recon_vector) / (np.linalg.norm(orig_vector) * np.linalg.norm(recon_vector)))
                            if not np.isnan(cos_sim):
                                pos_penalty[pos_token_index] = (cos_sim - 1) / 2
        
        return obj_penalty, att_penalty, rel_penalty, pos_penalty

    def __call__(
        self,
        prompt: str,
        tokenized_prompt: List[str],
        image: Image.Image,
        model_image: Image.Image,
    ):  
        rec_reward = np.zeros(len(tokenized_prompt))
        obj_penalty = np.zeros(len(tokenized_prompt))
        att_penalty = np.zeros(len(tokenized_prompt))
        rel_penalty = np.zeros(len(tokenized_prompt))
        pos_penalty = np.zeros(len(tokenized_prompt))

        image = image.convert("RGB")
        model_image = model_image.convert("RGB")

        tokenized_prompt = tokenized_prompt[:-1]  # remove </s>

        (
            obj_penalty[:len(tokenized_prompt)],
            att_penalty[:len(tokenized_prompt)],
            rel_penalty[:len(tokenized_prompt)],
            pos_penalty[:len(tokenized_prompt)],
        ) = self.get_scores(
            prompt,
            tokenized_prompt,
            image,
            model_image,
        )

        rec_reward[-1] = (get_cos_sim(image, model_image) + 1) / 2

        return rec_reward, obj_penalty, att_penalty, rel_penalty, pos_penalty
