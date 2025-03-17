import sys
import os
# Add the desired path
from .FastSAM.fastsam import FastSAM, FastSAMPrompt
import ast
import torch
from PIL import Image
from .FastSAM.utils.tools import convert_box_xywh_to_xyxy
import cv2
def fast_sam(
    model_path="/data/home/jian/RLS_microwave/utils/foundation_models/FastSAM/weights/FastSAM-x.pt",
        img_path=os.path.expanduser('~/TextToActions/dataset/_2_control_panel_images/_1_selected/2_air_purifier/0_0.jpeg'),
        imgsz=1024,
        iou=0.1,
        text_prompt=None,
        conf=0.9,
        output="./output/",
        randomcolor=True,
        point_prompt="[[0,0]]",
        point_label="[0]",
        box_prompt="[[0,0,0,0]]",
        better_quality=False,
        device=None,
        retina=True,
        withContours=False
):
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    # load model
    model = FastSAM(model_path)
    point_prompt = ast.literal_eval(point_prompt)
    box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(box_prompt))
    point_label = ast.literal_eval(point_label)
    input_image = Image.open(img_path).convert("RGB")
    
    # enhance contrast
    #input_image = cv2.imread(img_path)
    #gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    #input_image = cv2.adaptiveThreshold(
    #    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    #)

    everything_results = model(
        input_image,
        device=device,
        retina_masks=retina,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        #better_quality = better_quality
    )

    bboxes = []
    for result in everything_results:
        boxes = result.boxes.xyxyn.cpu().numpy().tolist()  # Bounding boxes in (x1, y1, x2, y2) format
        scores = result.boxes.conf.cpu().numpy().tolist()  # Confidence scores
        
        for box, score in zip(boxes, scores):
            dict_item = {
                "score": score,
                "bbox": box,
                "label": "sam"
            }
            bboxes.append(dict_item)

    return bboxes
    