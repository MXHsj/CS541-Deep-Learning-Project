# =======================================================================
# file name:    vis.py
# description:  utility functions for visualization
# authors:      Xihan Ma, Mingjie Zeng, Xiaofan Zhou
# date:         2022-11-13
# version:
# =======================================================================
import torch
from torchvision import transforms

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def array2tensor(array: np.array, device="cpu") -> torch.tensor:
  ''' convert input image in numpy array to tensor
  :param array:   input image (W x H)
  :param device:  "cpu" / "cuda"
  :return:        image in tensor that can be directly fed into network (1 x 1 x W x H)
  '''
  array = np.expand_dims(array, axis=0)
  array = np.expand_dims(array, axis=0)
  tensor = torch.from_numpy(array).to(device).type(torch.float32)
  return tensor


def tensor2array(tensor: torch.tensor, device="cpu") -> np.array:
  ''' convert input image in tensor to numpy array
  :param tensor:  input image (1 x 1 x W x H)
  :param device:  "cpu" / "cuda"
  :return:        image in numpy array that can be directly displayed (W x H)
  '''
  if device == "cpu":
    array = tensor.cpu().clone().detach().numpy()
  elif device == "cuda":
    array = tensor.clone().detach().numpy()
  array = np.squeeze(array)
  return array


def tensor2PIL(tensor: torch.tensor, device="cuda") -> Image:
  ''' convert input array in tensor to PIL Image
  :param:
  :param:
  :return: image in PIL Image that can be directly displayed (W x H)
  '''
  image = tensor.clone()  # do not specify cpu to avoid threading error when using gpu
  image = image.squeeze(0)
  image = transforms.ToPILImage()(image)
  return image


def plot_segmentation(epoch, global_step, image: Image, mask: Image, pred_mask: Image, showNow=False) -> None:
  # TODO:
  # - merge file name string as one input
  # -

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

  if showNow:
    plt.pause(0.001)

  plt.savefig(save_path + 'epoch_' + str(epoch) + '_step_' + str(global_step) + '.png')
  plt.close()
