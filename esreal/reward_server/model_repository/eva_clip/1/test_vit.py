from typing import List

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tqdm.auto import tqdm
from tritonclient.utils import np_to_triton_dtype
from utils import from_image_to_b64


def request_vit(images: List[Image.Image], triton_server_url="localhost:8000"):
    # images = [image.resize((512, 512)) for image in images]
    b64_images = np.array([[from_image_to_b64(image)] for image in images])
    # with open("images.txt", "w") as f:
    # f.write(str(b64_images[0][0]))

    with httpclient.InferenceServerClient(triton_server_url) as client:
        inputs = [
            httpclient.InferInput("IMAGE", b64_images.shape, np_to_triton_dtype(b64_images.dtype)),
        ]

        inputs[0].set_data_from_numpy(b64_images)

        outputs = [
            httpclient.InferRequestedOutput("CROP_IMAGE"),
            httpclient.InferRequestedOutput("IMAGE_EMBEDS"),
        ]

        response = client.infer("eva_clip", inputs, request_id=str(1), outputs=outputs)
        crop_image = response.as_numpy("CROP_IMAGE")  # (B, 224, 224, 3)
        image_embeds = response.as_numpy("IMAGE_EMBEDS")  # (B, 257, 1408)

    return crop_image, image_embeds


if __name__ == "__main__":
    triton_server_url = "dgx-11:8000"

    images = [Image.open("in.png")] * 8

    for idx in tqdm(range(1000)):
        crop_image, image_embeds = request_vit(images, triton_server_url=triton_server_url)
        if idx == 0:
            print(crop_image.shape, image_embeds.shape)
