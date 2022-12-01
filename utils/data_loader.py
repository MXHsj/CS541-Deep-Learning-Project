# =======================================================================
# file name:    data_loader.py
# description:  implement dataloader for pytorch
# authors:      Xihan Ma, Mingjie Zeng, Xiaofan Zhou
# date:         2022-11-30
# version:
# =======================================================================
import os
import cv2
import random
import torch
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset
from PIL import Image
from scipy.ndimage.interpolation import zoom


def random_rot_flip(image, label):
  k = np.random.randint(0, 4)
  image = np.rot90(image, k)
  label = np.rot90(label, k)
  axis = np.random.randint(0, 2)
  image = np.flip(image, axis=axis).copy()
  label = np.flip(label, axis=axis).copy()
  return image, label


def random_rotate(image, label):
  angle = np.random.randint(-10, 10)
  image = ndimage.rotate(image, angle, order=0, reshape=False)
  label = ndimage.rotate(label, angle, order=0, reshape=False)
  return image, label


def random_deform(image, label):
  # TODO: elastic deformation
  return image, label


class RandomGenerator(object):
  def __init__(self, output_size,encoder):
    self.output_size = output_size
    self.encoder = encoder
  def __call__(self, sample):
    image, label = sample['image'], sample['label']

    if random.random() > 0.5:
      image, label = random_rot_flip(image, label)
    elif random.random() > 0.5:
      image, label = random_rotate(image, label)
    x, y = image.shape
    if x != self.output_size[0] or y != self.output_size[1]:
      image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
      #label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
      if self.encoder:
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 3), order=3)
      else:
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 3), order=0)
    image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
    label = torch.from_numpy(label.astype(np.float32))
    sample = {'image': image, 'label': label.long()}
    return sample


class LUSDataset(Dataset):
  def __init__(self, use_patient_data=True,encoder=True):
    """ 
    :use_patient_data:
    """
    self.encoder=encoder
    if use_patient_data:
      if encoder:
        print('load patient data')
        self.img_dir = './dataset_patient_nolabel/image/'
        self.msk_dir = './dataset_patient_nolabel/mask/'
      else:
        self.img_dir = './dataset_patient/image/'
        self.msk_dir = './dataset_patient/mask/'
    self.sample_list=os.listdir(self.msk_dir)

    self.INPUT_HEIGHT = 224
    self.INPUT_WIDTH = 224

  def preprocess(self, frame, isMsk=False):
    x,y=frame.shape
    processed = zoom(frame, (self.INPUT_HEIGHT / x, self.INPUT_WIDTH / y), order=3)
    if isMsk:
      processed[processed > 2] = 0  # force two classes
    processed_np = processed.copy()
    if not isMsk:
      if processed_np.ndim == 2:
        processed_np = processed_np[np.newaxis, ...]
      else:
        processed_np = processed_np.transpose((2, 0, 1))
      processed_np = processed_np / 255  # normalize
    return processed_np

  def __len__(self):
    return len(os.listdir(self.msk_dir))

  def __getitem__(self, idx):
    # print(self.img_dir+img_names[idx])
    # print(self.msk_dir+msk_names[idx])
    msk = cv2.imread(self.msk_dir+self.sample_list[idx], cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(self.img_dir+self.sample_list[idx], cv2.IMREAD_GRAYSCALE)
    img = self.preprocess(img)
    if self.encoder:
      msk = self.preprocess(msk)
    else:
      msk = self.preprocess(msk, isMsk=True)
    #print(f'msk max val: {np.max(msk)}')

    return {
        'image': torch.as_tensor(img.copy()).float().contiguous(),
        'mask': torch.as_tensor(msk.copy()).long().contiguous()
    }

  def view_item(self, idx):
    ...
