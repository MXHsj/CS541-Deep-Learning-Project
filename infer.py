# =======================================================================
# file name:    infer.py
# description:
# authors:      Xihan Ma, Mingjie Zeng, Xiaofan Zhou
# date:         2022-11-13
# version:
# =======================================================================
# TODO: implement real-time inference in time-series

import cv2
import time
import torch
import argparse
import numpy as np
from unet.model import UNet
from utils.vis import array2tensor, tensor2array


# load network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")
net = UNet(n_channels=1, n_classes=3, bilinear=False)
net.to(device=device)
net.load_state_dict(torch.load('checkpoints/checkpoint_epoch5.pth'))
# print(net.eval())

# load example image
image = cv2.imread('dataset_patient/image/din-01-L-1-V-frame0.jpg', cv2.IMREAD_GRAYSCALE)
HEIGHT_ORIG = image.shape[1]
WIDTH_ORIG = image.shape[0]
image = cv2.resize(image, (128, 128))
image = array2tensor(image, device=device)
print(f'input shape: {image.shape}')

for _ in range(100):
  start = time.perf_counter()

  pred = net(image)
  pred = tensor2array(pred)
  print(f'prediction shape: {pred.shape}')
  pl = pred[1, :, :]*255  # separate pleural line mask
  rs = pred[2, :, :]*255  # separate rib shadow mask
  pl = cv2.resize(pl, (HEIGHT_ORIG, WIDTH_ORIG))
  rs = cv2.resize(rs, (HEIGHT_ORIG, WIDTH_ORIG))

  print(f'time elapsed: {time.perf_counter()-start} sec')  # benchmarking

# cv2.imwrite('test_pleural_line_out.png', pl)
# cv2.imwrite('test_rib_shadow_out.png', rs)
