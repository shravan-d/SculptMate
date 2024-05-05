import os

import numpy as np
from PIL import Image
# import cv2
import torch
import torchvision.transforms as transforms


load_size = 1024
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def read_image_from_path(img_path):
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if im.shape[2] == 4:
        im = im / 255.0
        im[:, :, :3] /= im[:, :, 3:] + 1e-8
        im = im[:, :, 3:] * im[:, :, :3] + 0.5 * (1.0 - im[:, :, 3:])
        im = (255.0 * im).astype(np.uint8)

    im = cv2.resize(im, (load_size, load_size))
    image = Image.fromarray(im[:, :, ::-1]).convert('RGB')

    return image, img_name


def get_pifu_input(image, scale, img_name="InputImage"):
    
    image_512 = image.copy()
    image_512.thumbnail([512, 512], Image.Resampling.LANCZOS)

    intrinsic = np.identity(4)
    trans_mat = np.identity(4)

    trans_mat *= scale
    trans_mat[3, 3] = 1.0
    trans_mat[0, 3] = 0  # -scale*(rect[0] + rect[2]//2 - w//2) * scale_im2ndc
    trans_mat[1, 3] = 0  # scale*(rect[1] + rect[3]//2 - h//2) * scale_im2ndc

    intrinsic = np.matmul(trans_mat, intrinsic)
    projection_matrix = np.identity(4)
    projection_matrix[1, 1] = -1
    calib = torch.Tensor(projection_matrix).float()

    calib_world = torch.Tensor(intrinsic).float()

    image_512 = transform(image_512)
    image = transform(image)
    return {
        'name': img_name,
        'img': image.unsqueeze(0),
        'img_512': image_512.unsqueeze(0),
        'calib': calib.unsqueeze(0),
        'calib_world': calib_world.unsqueeze(0),
        'b_min': np.array([-1, -1, -1]),
        'b_max': np.array([1, 1, 1]),
    }