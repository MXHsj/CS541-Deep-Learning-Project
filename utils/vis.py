# =======================================================================
# file name:    vis.py
# description:  utility functions for visualization
# authors:      Xihan Ma, Mingjie Zeng, Xiaofan Zhou
# date:         2022-11-13
# version:
# =======================================================================
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def tensor2PIL(tensor: torch.tensor) -> Image:
  image = tensor.clone()  # do not specify cpu to avoid threading error when using gpu
  image = image.squeeze(0)  # remove the fake batch dimension
  image = transforms.ToPILImage()(image)
  return image


def save_fig(epoch, global_step, image: Image, mask: Image, pred_mask: Image, isShow=False) -> None:
  save_path = './training_log/'
  # create figure
  fig = plt.figure(figsize=(10, 7))

  # setting values to rows and column variables
  rows = 1
  columns = 3

  # Adds a subplot at the 1st position
  fig.add_subplot(rows, columns, 1)

  # showing image
  plt.imshow(image, cmap='gray')
  plt.axis('off')
  plt.title("Image")

  # Adds a subplot at the 2nd position
  fig.add_subplot(rows, columns, 2)

  # showing image
  plt.imshow(mask, cmap='gray')
  plt.axis('off')
  plt.title("True mask")

  # Adds a subplot at the 3rd position
  fig.add_subplot(rows, columns, 3)

  # showing image
  plt.imshow(pred_mask, cmap='gray')
  plt.axis('off')
  plt.title("Pred_mask")

  if isShow:
    plt.pause(0.001)

  plt.savefig(save_path + 'epoch_' + str(epoch) + '_step_' + str(global_step) + '.png')
  plt.close()
