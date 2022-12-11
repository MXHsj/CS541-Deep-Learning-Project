import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2


def random_rot_flip(image, label):
    # TODO: 
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


class RandomGenerator(object):
    def __init__(self, output_size, encoder=False):
        self.output_size = output_size
        self.encoder=encoder

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        label[label > 3] = 0
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            #label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        if not self.encoder:
            sample = {'image': image, 'label': label.long()}
        else:
            sample = {'image': image, 'label': label}
        return sample


class Lung_dataset(Dataset):
    def __init__(self, base_dir, split='train', transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = os.listdir(base_dir + '/mask_merged')
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)
    def __getitem__(self, idx):
        if self.split == "train":
            path_image = self.data_dir + '/image/' + self.sample_list[idx]
            path_label = self.data_dir + '/mask_merged/' + self.sample_list[idx]
            image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(path_label, cv2.IMREAD_GRAYSCALE)
        else:
            path_image = self.data_dir + '/image/' + self.sample_list[idx]
            path_label = self.data_dir + '/mask_merged/' + self.sample_list[idx]
            image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(path_label, cv2.IMREAD_GRAYSCALE)
            image, label = image[:], label[:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
