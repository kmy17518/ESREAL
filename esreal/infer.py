import os
import gc
import json
from typing import Optional

import fire
import torch
import torch.distributed as dist

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator

from lavis.datasets.builders import load_dataset
from lavis.models import load_model_and_preprocess

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main(
    target_checkpoint: Optional[str],
    save_dir: str,
    save_filename: str,
    start_index: int,
    interval: int,
    dataset_name: str,
    df_path: str,
    image_dir: str,
    prompt: str,
    batch_size: int,
    num_workers: int,
):
    # create accelerator
    accelerator = Accelerator()

    # load model
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5_instruct_lora_with_value_head",
        model_type="flant5xl",
        is_eval=True,
    )

    if target_checkpoint is not None:
        model.load_state_dict(torch.load(target_checkpoint, map_location="cpu"), strict=False)

    # load tokenizer
    t5_tokenizer = model.t5_tokenizer
    tokenized = t5_tokenizer(prompt, return_tensors="pt")

    # load dataset
    dataset = load_dataset(dataset_name, df_path=df_path, image_dir=image_dir, include_pil=False)
    test_dataset = dataset["test"]
    max_size = len(test_dataset)
    accelerator.print(f"Total test dataset size: {len(test_dataset)}")
    test_dataset = Subset(test_dataset, range(start_index, min(start_index + interval, max_size)))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # prepare accelerator
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    # inference
    inference_results = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            image_ids = batch["image_id"].tolist()
            image_paths = batch["image_path"]
            images = [vis_processors["eval"](Image.open(image_path).convert("RGB")) for image_path in image_paths]
            images = torch.stack(images).to(accelerator.device)
            input_ids = tokenized.input_ids.repeat_interleave(images.shape[0], dim=0).to(accelerator.device)
            outputs = accelerator.unwrap_model(model).generate(
                images,
                input_ids,
                top_p=1.0,
            )
            output_texts = t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            inference_results.extend(
                [
                    {"image_id": image_id, "caption": output_text}
                    for image_id, output_text in zip(image_ids, output_texts)
                ]
            )

            torch.cuda.empty_cache()
            gc.collect()

    # gather results
    if accelerator.is_main_process:
        gathered_results = [None] * dist.get_world_size()
        dist.gather_object(inference_results, gathered_results)
    else:
        dist.gather_object(inference_results)

    # save results
    if accelerator.is_main_process:
        # reformat gathered_results
        final_results = []
        for res in gathered_results:
            final_results.extend(res)
        if interval < BIG_NUMBER:
            save_filename = save_filename.replace(".jsonl", f"__{start_index}__{start_index + interval}.jsonl")
        save_path = os.path.join(save_dir, save_filename)
        with open(save_path, "w") as f:
            for item in final_results:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")


if __name__ == "__main__":
    fire.Fire(main)
