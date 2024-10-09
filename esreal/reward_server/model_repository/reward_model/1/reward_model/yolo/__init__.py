from typing import List

import numpy as np
import ultralytics
from PIL import Image

from ..registry import registry


@registry.register("yolo")
class YOLO:
    def __init__(self, model_name: str = "yolov8x.pt"):
        self.model = ultralytics.YOLO(model_name)

    def __call__(self, images: List[Image.Image], debug: bool = False) -> List[ultralytics.engine.results.Results]:
        results = self.model(images)
        bboxes = []
        for result in results:
            if debug:
                im_array = result.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                im.save("out_detect.jpg")  # save image

            boxes = result.boxes.cpu().numpy()
            bboxes.append((boxes.xyxy.tolist(), [self.model.names[cls] for cls in boxes.cls]))
        return np.array(bboxes, dtype=np.object_)
