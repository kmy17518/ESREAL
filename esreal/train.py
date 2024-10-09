import os
import argparse
import functools
import time
from copy import deepcopy
from typing import List

import torch
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

from lavis.datasets.builders import load_dataset
from lavis.models import load_model_and_preprocess

import trlx
from trlx.data.default_configs import (
    TRLConfig,
    TrainConfig,
    TokenizerConfig,
    OptimizerConfig,
    SchedulerConfig,
    PPOConfig,
)
from trlx.data.configs import NNModelConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function '{func.__name__}' took: {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def default_ppo_config(model, model_copy, args):
    return TRLConfig(
        train=TrainConfig(
            seq_length=args.seq_length,
            epochs=100,
            total_steps=10000,
            batch_size=args.batch_size,
            checkpoint_interval=100
            eval_interval=100,
            pipeline="VLMDatasetPipeline",
            trainer="AccelerateVLMPPOTrainer",
            checkpoint_dir=args.checkpoint_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
        ),
        model=NNModelConfig(
            model_arch_type="seq2seq",
            ref_to_model=model,
            model_copy=model_copy,
            model_init_kwargs=dict(),
        ),
        tokenizer=TokenizerConfig(
            tokenizer_path=args.model_path,
            padding_side="left",
            truncation_side="right",
        ), 
        optimizer=OptimizerConfig(
            name="adamw",
            kwargs=dict(lr=args.lr, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6),
        ),
        scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=3e-5)
        ),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=args.chunk_size,
            ppo_epochs=4,
            init_kl_coef=args.init_kl_coef,
            target=None,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
        ),
    )


def create_reference_model(model):
    parameter_names = [n for n, _ in model.named_parameters()]
    ref_model = deepcopy(model)
    for param_name in parameter_names:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False
    return ref_model.eval()


def single_request_reward_model(
        images: np.ndarray,
        prompt: np.ndarray,
        tokenized_prompt: List[str],
        triton_server_url: str,
    ):
    tokenized_prompt = np.array(tokenized_prompt, dtype=np.object_)[None, :]  # (1, T)

    with httpclient.InferenceServerClient(triton_server_url, network_timeout=600) as client:
        inputs = [
            httpclient.InferInput("IMAGE", images.shape, np_to_triton_dtype(images.dtype)),
            httpclient.InferInput("PROMPT", prompt.shape, np_to_triton_dtype(prompt.dtype)),
            httpclient.InferInput("TOKENIZED_PROMPT", tokenized_prompt.shape, np_to_triton_dtype(tokenized_prompt.dtype)),
        ]
        inputs[0].set_data_from_numpy(images)
        inputs[1].set_data_from_numpy(prompt)
        inputs[2].set_data_from_numpy(tokenized_prompt)

        outputs = [
            httpclient.InferRequestedOutput("MEAN_REC_REWARD"),
            httpclient.InferRequestedOutput("MEAN_OBJ_PENALTY"),
            httpclient.InferRequestedOutput("MEAN_ATT_PENALTY"),
            httpclient.InferRequestedOutput("MEAN_REL_PENALTY"),
            httpclient.InferRequestedOutput("MEAN_POS_PENALTY"),
        ]
        max_retries = 5
        for attempt in range(max_retries):
            try:
                result = client.infer("reward_model", inputs, request_id=str(1), outputs=outputs)
                mean_rec_reward = result.as_numpy("MEAN_REC_REWARD").squeeze().tolist()
                mean_obj_penalty = result.as_numpy("MEAN_OBJ_PENALTY").squeeze().tolist()
                mean_att_penalty = result.as_numpy("MEAN_ATT_PENALTY").squeeze().tolist()
                mean_rel_penalty = result.as_numpy("MEAN_REL_PENALTY").squeeze().tolist()
                mean_pos_penalty = result.as_numpy("MEAN_POS_PENALTY").squeeze().tolist()
                return mean_rec_reward, mean_obj_penalty, mean_att_penalty, mean_rel_penalty, mean_pos_penalty
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f'Retrying due to error: {e}')
                    continue
                else:
                    raise e
    return mean_rec_reward, mean_obj_penalty, mean_att_penalty, mean_rel_penalty, mean_pos_penalty


def request_reward_model(
    total_images: np.array,
    total_prompts: List[str],
    total_tokenized_prompts: List[List[str]],
    triton_server_url: str,
):
    if getattr(request_reward_model, "pool", None) is None:
        from multiprocessing.pool import Pool
        request_reward_model.pool = Pool(16)
    pool = request_reward_model.pool
    total_images = total_images[:, None]  # (N, 1, 3, 224, 224)
    total_prompts = np.array(total_prompts, dtype=np.object_)[:, None][:, None]  # (N, 1, 1)

    mean_rec_reward, mean_obj_penalty, mean_att_penalty, mean_rel_penalty, mean_pos_penalty = zip(
        *pool.starmap(
            single_request_reward_model, zip(
                total_images,
                total_prompts,
                total_tokenized_prompts,
                [triton_server_url] * len(total_images),
            )
        )
    )
    return mean_rec_reward, mean_obj_penalty, mean_att_penalty, mean_rel_penalty, mean_pos_penalty


def main(args):
    dataset_name = args.dataset_name
    triton_server_url = args.triton_server_url
    alpha = args.alpha
    is_rec_penalty = args.is_rec_penalty

    device = int(os.environ.get("LOCAL_RANK", 0)) if torch.cuda.is_available() else -1

    model, _, _ = load_model_and_preprocess(
        name="blip2_t5_instruct_lora_with_value_head",
        model_type="flant5xl",
        is_eval=False,
        device=device,
    )
    model_copy = create_reference_model(model)

    config = default_ppo_config(model, model_copy, args)

    dataset = load_dataset(dataset_name, df_path=args.df_path, image_dir=args.image_dir)


    @timeit
    def dense_reward_fn(
        images: torch.Tensor,
        samples: List[str],
        prompts: List[str],
        outputs: List[str],
        tokenizer,
        **kwargs
    ) -> List[float]:
        images = images.detach().cpu().numpy()
        outputs = [str(output) for output in outputs]
        tokenized_outputs = [[tokenizer.decode(token) for token in tokenizer(output).input_ids] for output in outputs]

        (
            mean_rec_reward,
            mean_obj_penalty,
            mean_att_penalty,
            mean_rel_penalty,
            mean_pos_penalty,
        ) = request_reward_model(
            images,
            outputs,
            tokenized_outputs,
            triton_server_url,
        )

        B = len(images)
        dense_rewards = []

        for i in range(B):
            L = len(tokenized_outputs[i])
            dense_reward = []

            for j in range(L):
                if is_rec_penalty:
                    dense_reward.append(
                        (
                            (0 if mean_rec_reward[i][j] == 0 else alpha * (mean_rec_reward[i][j] - 1))
                            + mean_obj_penalty[i][j]
                            + mean_att_penalty[i][j]
                            + mean_rel_penalty[i][j]
                            + mean_pos_penalty[i][j]
                        )
                    )
                else:
                    dense_reward.append(
                        (
                            alpha * mean_rec_reward[i][j]
                            + mean_obj_penalty[i][j]
                            + mean_att_penalty[i][j]
                            + mean_pos_penalty[i][j]
                        )
                    )

            dense_rewards.append(dense_reward)

        reward_dict = {
            "dense_rewards": dense_rewards,
            "rec_reward": mean_rec_reward,
            "obj_penalty": mean_obj_penalty,
            "att_penalty": mean_att_penalty,
            "rel_penalty": mean_rel_penalty,
            "pos_penalty": mean_pos_penalty,
        }

        return dense_rewards, reward_dict


    trlx.train(
        reward_fn=dense_reward_fn,
        prompts=dataset["train"],
        eval_prompts=dataset["val"],
        config=config,
        triton_server_url=triton_server_url,
        args=args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--triton_server_url', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--df_path', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--task_name', type=str, choices=["short_caption", "long_caption", "vqa"])
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--chunk_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--seq_length', type=int)
    parser.add_argument('--max_new_tokens', type=int)
    parser.add_argument('--repetition_penalty', type=float)
    parser.add_argument('--init_kl_coef', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--is_rec_penalty', type=bool)
    parser.add_argument('--checkpoint_dir', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--resume_from_checkpoint', type=str)
    args = parser.parse_args()

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)

    main(args)
