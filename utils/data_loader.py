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
from torch.utils.data import Dataset
from PIL import Image
# from scipy.ndimage.interpolation import zoom


def random_rot_flip(image, label):
  k = np.random.randint(0, 4)
  image = np.rot90(image, k)
  label = np.rot90(label, k)
  axis = np.random.randint(0, 2)
  image = np.flip(image, axis=axis).copy()
  label = np.flip(label, axis=axis).copy()
  return image, label


def random_rotate(image, label):
  angle = np.random.randint(-20, 20)
  image = ndimage.rotate(image, angle, order=0, reshape=False)
  label = ndimage.rotate(label, angle, order=0, reshape=False)
  return image, label


# class RandomGenerator(object):
#   def __init__(self, output_size):
#     self.output_size = output_size

#   def __call__(self, sample):
#     image, label = sample['image'], sample['label']

#     if random.random() > 0.5:
#       image, label = random_rot_flip(image, label)
#     elif random.random() > 0.5:
#       image, label = random_rotate(image, label)
#     x, y = image.shape
#     if x != self.output_size[0] or y != self.output_size[1]:
#       image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
#       #label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
#       label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 3), order=0)
#     image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
#     label = torch.from_numpy(label.astype(np.float32))
#     sample = {'image': image, 'label': label.long()}
#     return sample


class LUSDataset(Dataset):
  def __init__(self, use_patient_data=True):
    """ 
    :use_patient_data:
    """
    if use_patient_data:
      print('load patient data')
      self.img_dir = os.path.join(os.path.dirname(__file__), '../dataset_patient/image/')
      self.msk_dir = os.path.join(os.path.dirname(__file__), '../dataset_patient/mask_merged/')
    assert (len(os.listdir(self.img_dir)) == len(os.listdir(self.msk_dir)))

    self.INPUT_HEIGHT = 128
    self.INPUT_WIDTH = 128

  def preprocess(self, frame, isMsk=False):
    processed = cv2.resize(frame, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
    # processed = frame.resize((self.INPUT_WIDTH, self.INPUT_HEIGHT), resample=Image.NEAREST if isMsk else Image.BICUBIC)

    if isMsk:
      processed[processed > 2] = 0  # force two classes
      # print(f"mask: {processed.size}")
      print(f"mask: {processed.shape}")

    # print(f"img: {processed.size}")

    # processed_np = np.asarray(processed)
    processed_np = processed.copy()

    print(f"np_mask: {processed_np.shape}")

    if not isMsk:
      if processed_np.ndim == 2:
        processed_np = processed_np[np.newaxis, ...]
      else:
        processed_np = processed_np.transpose((2, 0, 1))

      processed_np = processed_np / 255
      print(f"np_img: {processed_np.shape}")

    return processed_np

  def __len__(self):
    return len(os.listdir(self.img_dir))

  def __getitem__(self, idx):
    img_names = os.listdir(self.img_dir)
    msk_names = os.listdir(self.msk_dir)
    print(self.img_dir+img_names[idx])
    print(self.msk_dir+msk_names[idx])
    msk = cv2.imread(self.msk_dir+msk_names[idx], cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread(self.img_dir+img_names[idx], cv2.IMREAD_COLOR)  # TODO: use grayscale input

    img = cv2.imread(self.img_dir+img_names[idx], cv2.IMREAD_GRAYSCALE)
    # msk = Image.open(self.img_dir+msk_names[idx]).convert('L')
    # img = Image.open(self.img_dir+img_names[idx])
    print("Image size ------------")
    #print(f"size of img: {img.size}")
    #print(f"size of mask: {msk.size}")

    #img = cv2.imread(self.img_dir+img_names[idx])
    #msk = cv2.imread(self.img_dir+msk_names[idx])
    #print("cv2 size------------")
    #print(f"size of img: {img[0].shape}")
    #print(f"size of mask: {msk[0].shape}")

    # assert (np.all(img.shape == msk.shape))
    # assert (np.all(img.size == msk.size))
    img = self.preprocess(img)
    msk = self.preprocess(msk, isMsk=True)
    print(f'msk max val: {np.max(msk)}')

    return {
        'image': torch.as_tensor(img.copy()).float().contiguous(),
        'mask': torch.as_tensor(msk.copy()).long().contiguous()
    }

  def view_item(self, idx):
    ...


# if __name__ == '__main__':
#  # test dataloader
#  dataset = LUSDataset()
#  dataset[0]
