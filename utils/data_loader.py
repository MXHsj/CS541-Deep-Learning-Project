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
import elasticdeform
from scipy import ndimage
from scipy.ndimage import zoom

from torch.utils.data import Dataset

ORIG_HEIGHT = 820
ORIG_WIDTH = 1124
# INPUT_HEIGHT = 224
# INPUT_WIDTH = 224
INPUT_HEIGHT = round(ORIG_HEIGHT/4)
INPUT_WIDTH = round(ORIG_WIDTH/4)


def _normalize_inputs(X):
  if isinstance(X, np.ndarray):
    Xs = [X]
  elif isinstance(X, list):
    Xs = X
  else:
    raise Exception('X should be a numpy.ndarray or a list of numpy.ndarrays.')

  # check X inputs
  assert len(Xs) > 0, 'You must provide at least one image.'
  assert all(isinstance(x, np.ndarray) for x in Xs), 'All elements of X should be numpy.ndarrays.'
  return Xs


def _normalize_axis_list(axis, Xs):
  if axis is None:
    axis = [tuple(range(x.ndim)) for x in Xs]
  elif isinstance(axis, int):
    axis = (axis,)
  if isinstance(axis, tuple):
    axis = [axis] * len(Xs)
  assert len(axis) == len(Xs), 'Number of axis tuples should match number of inputs.'
  input_shapes = []
  for x, ax in zip(Xs, axis):
    assert isinstance(ax, tuple), 'axis should be given as a tuple'
    assert all(isinstance(a, int) for a in ax), 'axis must contain ints'
    assert len(ax) == len(axis[0]), 'All axis tuples should have the same length.'
    assert ax == tuple(set(ax)), 'axis must be sorted and unique'
    assert all(0 <= a < x.ndim for a in ax), 'invalid axis for input'
    input_shapes.append(tuple(x.shape[d] for d in ax))
  assert len(set(input_shapes)) == 1, 'All inputs should have the same shape.'
  deform_shape = input_shapes[0]
  return axis, deform_shape


def random_rot_flip(image: np.ndarray, label: np.ndarray):
  image = np.flip(image, axis=1).copy()  # horizontal flip
  label = np.flip(label, axis=1).copy()
  return image, label


def random_rotate(image: np.ndarray, label: np.ndarray):
  angle = np.random.randint(-10, 10)
  image = ndimage.rotate(image, angle, order=0, reshape=False)
  label = ndimage.rotate(label, angle, order=0, reshape=False)
  return image, label


def random_deform(image: np.ndarray, label: np.ndarray):
  ''' perform random elastic deformation using 3x3 grid
  '''
  # image = elasticdeform.deform_random_grid(image, sigma=60, rotate=-5.05, points=3)
  Xs = _normalize_inputs(image)
  axis, deform_shape = _normalize_axis_list(None, Xs)
  sigma = 5
  points = 3
  if not isinstance(points, (list, tuple)):
    points = [points] * len(deform_shape)
  displacement = np.random.randn(len(deform_shape), *points) * sigma
  image = elasticdeform.deform_grid(image, displacement, axis=axis, rotate=-3.0)
  label = elasticdeform.deform_grid(label, displacement, axis=axis, rotate=-3.0)
  return image, label


class RandomGenerator(object):
  def __init__(self):
    pass

  def __call__(self, image, label):
    output_size = image.shape
    if random.random() > 0.5:
      image, label = random_rot_flip(image, label)
    if random.random() > 0.5:
      image, label = random_rotate(image, label)
    # if random.random() > 0.2:
    #   image, label = random_deform(image, label)
    x, y = image.shape
    if x != output_size[0] or y != output_size[1]:
      image = zoom(image, (output_size[0] / x, output_size[1] / y), order=3)
      label = zoom(label, (output_size[0] / x, output_size[1] / y), order=3)
    return image, label


class LUSDataset(Dataset):
  def __init__(self, use_patient_data=True, encoder=True, transform=None):
    """ 
    :use_patient_data:
    """
    self.encoder = encoder
    self.transform = transform
    if use_patient_data:
      if encoder:
        self.img_dir = os.path.dirname(__file__) + '/../dataset_patient_nolabel/image/'
        self.msk_dir = os.path.dirname(__file__) + '/../dataset_patient_nolabel/image/'
      else:
        self.img_dir = os.path.dirname(__file__) + '/../dataset_patient/image/'
        self.msk_dir = os.path.dirname(__file__) + '/../dataset_patient/mask_merged/'
    self.sample_list = os.listdir(self.msk_dir)

  def __len__(self):
    return len(os.listdir(self.msk_dir))

  def __getitem__(self, idx):
    img = cv2.imread(self.img_dir+self.sample_list[idx], cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(self.msk_dir+self.sample_list[idx], cv2.IMREAD_GRAYSCALE)
    if self.encoder:
      img = self.preprocess(img)
      msk = self.preprocess(msk)
      sample = {
          'image': torch.as_tensor(img.copy()).float().contiguous(),
          'mask': torch.as_tensor(msk.copy()).float().contiguous()  # use same type
      }
    else:
      if self.transform is not None:
        img, msk = self.transform(img, msk)
      img = self.preprocess(img)
      msk = self.preprocess(msk, isMsk=True)
      sample = {
          'image': torch.as_tensor(img.copy()).float().contiguous(),
          'mask': torch.as_tensor(msk.copy()).long().contiguous()
      }
    return sample

  def preprocess(self, frame, isMsk=False):
    ''' preprocess input image and mask
    '''
    processed = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    # x, y = frame.shape
    # processed = zoom(frame, (self.INPUT_HEIGHT / x, self.INPUT_WIDTH / y), order=3)

    if isMsk:
      processed[processed > 2] = 0  # force two classes
    if not isMsk:
      if processed.ndim == 2:
        processed = processed[np.newaxis, ...]
      else:
        processed = processed.transpose((2, 0, 1))
      processed = processed / 255  # normalize
    return processed

  def view_item(self, idx):
    ...
