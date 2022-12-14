# =======================================================================
# file name:    infer.py
# description:  real-time inference
# authors:      Xihan Ma, Mingjie Zeng, Xiaofan Zhou
# date:         2022-11-13
# version:
# =======================================================================

import cv2
import time
import torch
import numpy as np
from unet.model import UNet
import torch.nn.functional as F
from utils.vis import array2tensor, tensor2array


def get_pleural_area(img):
  return (img[:300, 238:880])


# load network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

# rib shadow model
net_dark = UNet(n_channels=1, n_classes=2, bilinear=False)
net_dark.to(device=device)
net_dark.load_state_dict(torch.load('checkpoints/feat0_checkpoint_epoch40.pth'))

# pleural line model
net_bright = UNet(n_channels=1, n_classes=2, bilinear=False)
net_bright.to(device=device)
net_bright.load_state_dict(torch.load('checkpoints/feat1_checkpoint_epoch40.pth'))

# load example image
image_raw = cv2.imread('dataset_patient/image/din-01-L-1-V-frame0.jpg', cv2.IMREAD_GRAYSCALE)
HEIGHT_ORIG = image_raw.shape[1]
WIDTH_ORIG = image_raw.shape[0]
cropped_raw = get_pleural_area(image_raw)

image = cv2.resize(image_raw, (128, 128))
image = array2tensor(image, device=device)

cropped = cv2.resize(cropped_raw, (128, 128))
cropped = array2tensor(cropped, device=device)
print(f'input shape: {image.shape}')

for _ in range(100):
  start = time.perf_counter()

  # ===== emulate real-time behavior =====
  cropped_raw = get_pleural_area(image_raw)
  cropped = cv2.resize(cropped_raw, (128, 128))
  cropped = array2tensor(cropped_raw, device=device)
  # ======================================

  pred_dark = net_dark(image)
  pred_bright = net_bright(cropped)
  pred_dark = tensor2array(pred_dark)
  pred_bright = tensor2array(pred_bright)
  # print(f'prediction shape: {pred_dark.shape}')

  pred_dark = cv2.resize(pred_dark[0, :, :]*255, (HEIGHT_ORIG, WIDTH_ORIG))
  pred_bright = cv2.resize(pred_bright[0, :, :]*255, (HEIGHT_ORIG, WIDTH_ORIG))

  print(f'time elapsed: {time.perf_counter()-start} sec')  # benchmarking

cv2.imwrite('test_pleural_line_out.png', pred_dark)
cv2.imwrite('test_rib_shadow_out.png', pred_bright)
