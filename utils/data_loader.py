# =======================================================================
# file name:    data_loader.py
# description:  implement dataloader for pytorch
# authors:      Xihan Ma, Mingjie Zeng, Xiaofan Zhou
# date:         2022-11-13
# version:
# =======================================================================
import os
import cv2
import random
import torch
import numpy as np
from scipy import ndimage
from PIL import Image
from scipy.ndimage.interpolation import zoom

from torch.utils.data import Dataset
from torchvision import transforms

from vis import tensor2array

INPUT_HEIGHT = 128
INPUT_WIDTH = 128
ORIG_HEIGHT = 820
ORIG_WIDTH = 1124


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


class RandomGenerator():
  def __init__(self):
    pass

  def __call__(self, sample):
    image, label = sample['image'], sample['mask']
    image = tensor2array(image)
    label = tensor2array(label)
    output_size = image.shape
    # force either random flipping or random rotation
    if random.random() > 0.5:
      image, label = random_rot_flip(image, label)
    else:
      image, label = random_rotate(image, label)
    x, y = image.shape
    if x != output_size[0] or y != output_size[1]:
      image = zoom(image, (output_size[0] / x, output_size[1] / y), order=3)
      label = zoom(label, (output_size[0] / x, output_size[1] / y), order=3)
    return {'image': image, 'mask': label}


class LUSDataset(Dataset):
  def __init__(self, sizefit2model=True, use_augmented_data=False):
    """ 
    :param fit2model:             shrink image to fit model input size
    :param use_patient_data:
    TODO:
    - use larger input image size
    """
    self.sizefit2model = sizefit2model
    if use_augmented_data:
      print('load augmented patient data')
      self.img_dir = os.path.join(os.path.dirname(__file__), '../dataset_patient/image_aug/')
      self.msk_dir = os.path.join(os.path.dirname(__file__), '../dataset_patient/mask_merged_aug/')
    else:
      print('load patient data')
      self.img_dir = os.path.join(os.path.dirname(__file__), '../dataset_patient/image/')
      self.msk_dir = os.path.join(os.path.dirname(__file__), '../dataset_patient/mask_merged/')
    assert (len(os.listdir(self.img_dir)) == len(os.listdir(self.msk_dir)))

  def __len__(self):
    return len(os.listdir(self.img_dir))

  def __getitem__(self, idx):
    img_names = os.listdir(self.img_dir)
    msk_names = os.listdir(self.msk_dir)
    # print(self.img_dir+img_names[idx])
    # print(self.msk_dir+msk_names[idx])
    msk = cv2.imread(self.msk_dir+msk_names[idx], cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(self.img_dir+img_names[idx], cv2.IMREAD_GRAYSCALE)
    # print(f"dim of img: {img.shape}")
    # print(f"dim of mask: {msk.shape}")

    # assert (np.all(img.shape == msk.shape))
    # assert (np.all(img.size == msk.size))
    img = self.preprocess(img)              # single channel, grey scale
    msk = self.preprocess(msk, isMsk=True)  # single channel, multiple labels
    # print(f'msk max val: {np.max(msk)}')

    return {
        'image': torch.as_tensor(img.copy()).float().contiguous(),
        'mask': torch.as_tensor(msk.copy()).long().contiguous()
    }

  def preprocess(self, frame: np.ndarray, isMsk=False):
    ''' preprocess input image and mask
    '''
    if self.sizefit2model:
      processed = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    else:
      processed = frame.copy()

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

  def view_item(self, idx):
    ...


if __name__ == '__main__':
  # ========== training data augmentation (only needs to run once per dataset) ==========
  img_aug_dir = os.path.join(os.path.dirname(__file__), '../dataset_patient/image_aug/')
  msk_aug_dir = os.path.join(os.path.dirname(__file__), '../dataset_patient/mask_merged_aug/')
  dataset = LUSDataset(sizefit2model=False, use_augmented_data=False)
  random_generator = RandomGenerator()
  print(f'size of training set: {len(dataset)}')
  aug_itr = 1
  for idx, item in enumerate(dataset):
    # ===== wirte original image and mask =====
    cv2.imwrite(img_aug_dir+f'frame{idx}.jpg', 255*tensor2array(item['image']))
    cv2.imwrite(msk_aug_dir+f'frame{idx}.jpg', tensor2array(item['mask']))
    # ===== write augmented image and mask ======
    for i in range(aug_itr):
      augmented1 = random_generator(item)  # first augmentation
      cv2.imwrite(img_aug_dir+f'frame{idx}_aug{i}.jpg', 255*augmented1['image'])
      cv2.imwrite(msk_aug_dir+f'frame{idx}_aug{i}.jpg', augmented1['mask'])
      print(f'data augmentation: {idx+1}/{len(dataset)}')
  print('finished')
