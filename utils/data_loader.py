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
import elasticdeform
from scipy import ndimage
from scipy.ndimage import zoom

from torch.utils.data import Dataset

INPUT_HEIGHT = 224  # 128
INPUT_WIDTH = 224  # 128
ORIG_HEIGHT = 820
ORIG_WIDTH = 1124


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


def get_pleural_area(img):
  return (img[:300, 238:880])


class RandomGenerator():
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
    # print('augmentation applied')
    x, y = image.shape
    if x != output_size[0] or y != output_size[1]:
      image = zoom(image, (output_size[0] / x, output_size[1] / y), order=3)
      label = zoom(label, (output_size[0] / x, output_size[1] / y), order=3)
    return image, label


class LUSDataset(Dataset):
  def __init__(self, trainID=0, transform=None):
    """ 
    :param:
    :param trainID: 0 -> train rib shadow; 1 -> train pleural line
    """
    self.trainID = trainID
    self.transform = transform

    self.img_dir = os.path.join(os.path.dirname(__file__), '../dataset_patient/image/')
    if self.trainID == 0:
      self.msk_dir = os.path.join(os.path.dirname(__file__), '../dataset_patient/mask/rib_shadow/')
    elif self.trainID == 1:
      self.msk_dir = os.path.join(os.path.dirname(__file__), '../dataset_patient/mask/pleural_line/')
    # assert (len(os.listdir(self.img_dir)) == len(os.listdir(self.msk_dir)))

  def __len__(self):
    return len(os.listdir(self.img_dir))

  def __getitem__(self, idx):
    img_names = os.listdir(self.img_dir)
    msk_names = os.listdir(self.msk_dir)
    img_names.sort()
    msk_names.sort()
    img = cv2.imread(self.img_dir+img_names[idx], cv2.IMREAD_GRAYSCALE)
    # TODO: skip training if no rib shadow mask exists
    try:
      msk = cv2.imread(self.msk_dir+msk_names[idx], cv2.IMREAD_GRAYSCALE)
    except Exception as e:
      print(f'loading mask: no mask\n')
      msk = np.zeros_like(img)
    # print(f"img dir: {self.img_dir+img_names[idx]}")
    # print(f"msk dir: {self.msk_dir+msk_names[idx]}")
    if self.transform is not None:
      img, msk = self.transform(img, msk)

    # assert (np.all(img.shape == msk.shape))
    # assert (np.all(img.size == msk.size))
    img = self.preprocess(img)              # single channel, grey scale
    msk = self.preprocess(msk, isMsk=True)  # single channel, multiple labels

    return {'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(msk.copy()).long().contiguous()}

  def preprocess(self, frame: np.ndarray, isMsk=False):
    ''' preprocess input image and mask
    '''
    if self.trainID == 1:
      processed = get_pleural_area(frame)  # crop image
    else:
      processed = frame.copy()
    # processed = frame.copy()
    processed = cv2.resize(processed, (INPUT_WIDTH, INPUT_HEIGHT))
    processed = processed / 255

    if isMsk:
      processed[processed > 0.2] = 1  # limit class label range
    else:
      if processed.ndim == 2:
        processed = processed[np.newaxis, ...]
      else:
        processed = processed.transpose((2, 0, 1))
    return processed

  def view_item(self, idx):
    ...
