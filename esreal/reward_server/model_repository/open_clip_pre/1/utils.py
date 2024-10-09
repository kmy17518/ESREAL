import base64
import io
import time
from contextlib import contextmanager

import cv2
import numpy as np
from PIL import Image


@contextmanager
def time_measure(label=""):
    start = time.time()
    yield
    end = time.time()
    print(f"[{label}] Time elapsed: {end - start}")


def from_image_to_b64(img: Image.Image) -> bytes:
    image_arr = np.array(img)[..., ::-1]
    _, imgByteArr = cv2.imencode(".png", image_arr)
    encoded = base64.b64encode(imgByteArr)
    return encoded


def from_b64_to_image(image_b64: str) -> Image.Image:
    image_byte = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_byte)).convert("RGB")
    return image
