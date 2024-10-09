import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class ImageParagraphDataset(Dataset):
    def __init__(self, df_path, image_dir, split="train", include_pil=True):
        raw_df = pd.read_csv(df_path)
        self.df = raw_df[raw_df[split]].reset_index()
        self.image_dir = image_dir
        self.include_pil = include_pil

    def __len__(self):
        return len(self.df)

    def collater(self, samples):
        return default_collate(samples)

    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]["Image_name"]
        image_path = os.path.join(self.image_dir, f"{image_name}.jpg")
        image = Image.open(image_path)

        text_input = "Write a detailed description for the image."
        text_output = self.df.iloc[idx]["Paragraph"]

        if self.include_pil:
            return {
                "image_id": image_name,
                "image": image,
                "text_input": text_input,
                "text_output": text_output,
            }
        else:
            return {
                "image_id": image_name,
                "image_path": image_path,
                "text_input": text_input,
                "text_output": text_output,
            }
