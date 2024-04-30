from segment_anything import sam_model_registry, SamPredictor
import os
import numpy as np
import torch
import cv2
import time
from PIL import Image

image_size = (1024, 1024)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_sam_model(device):
   sam_checkpoint = ROOT_DIR + "/checkpoints/sam_vit_h_4b8939.pth"
   model_type = "vit_h"

   sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
   return sam


def sam_out_nosave(predictor, input_image, bbox):
   bbox = np.array(bbox)
   image = np.asarray(input_image)

   start_time = time.time()
   predictor.set_image(image)

   masks_bbox, scores_bbox, logits_bbox = predictor.predict(
       box=bbox,
       multimask_output=True
   )
   
   out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
   out_image[:, :, :3] = image
   out_image_bbox = out_image.copy()
   out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255 # np.argmax(scores_bbox)
   torch.cuda.empty_cache()
   return Image.fromarray(out_image_bbox, mode='RGBA')


def image_preprocess_nosave(input_image, lower_contrast=True, rescale=True):

   image_arr = np.array(input_image)
   in_w, in_h = image_arr.shape[:2]

   if lower_contrast:
       alpha = 0.8  # Contrast control (1.0-3.0)
       beta =  0   # Brightness control (0-100)
       # Apply the contrast adjustment
       image_arr = cv2.convertScaleAbs(image_arr, alpha=alpha, beta=beta)
       image_arr[image_arr[...,-1]>200, -1] = 255

   ret, mask = cv2.threshold(np.array(input_image.split()[-1]), 1, 255, cv2.THRESH_BINARY)
   x, y, w, h = cv2.boundingRect(mask)
   max_size = max(w, h)
   ratio = 0.75
   if rescale:
       side_len = int(max_size / ratio)
   else:
       side_len = in_w
   scale = in_w / w
   padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
   center = side_len//2
   padded_image[center-h//2:center-h//2+h, center-w//2:center-w//2+w] = image_arr[y:y+h, x:x+w]
   rgba = Image.fromarray(padded_image).resize(image_size, Image.LANCZOS)

   rgba_arr = np.array(rgba) / 255.0
   rgb = rgba_arr[...,:3] * rgba_arr[...,-1:] + (1 - rgba_arr[...,-1:])
   return Image.fromarray((rgb * 255).astype(np.uint8)), scale


def preprocess_image(img_path, device):
    input_raw = Image.open(img_path)
    input_raw.thumbnail(image_size, Image.Resampling.LANCZOS)

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=ROOT_DIR+"/checkpoints/yolov5s.pt", force_reload=True) 
    results = model(input_raw)
    output = results.pandas().xyxy[0]
    objects = output[np.logical_and(output['name'] == 'person', output['confidence'] > 0.8)]
    bbox = [int(objects.xmin.iloc[0]), int(objects.ymin.iloc[0]), int(objects.xmax.iloc[0]), int(objects.ymax.iloc[0])]

    sam_model = get_sam_model(device)
    sam_predictor = SamPredictor(sam_model)

    image_sam = sam_out_nosave(sam_predictor, input_raw.convert("RGB"), bbox)

    input_image, scale = image_preprocess_nosave(image_sam, lower_contrast=True, rescale=True)
    return input_image, scale
