import argparse
import logging
import os
import random
import numpy as np
import torch
# import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse
import cv2
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (
    int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device=device)
net.load_from(weights=np.load(config_vit.pretrained_path))
test = cv2.imread('../input/LUS_patient_baseline/dataset/1/din-01-L-1-V-frame0.jpg',cv2.IMREAD_GRAYSCALE)
x, y = test.shape
test=zoom(test, (224 / x, 224 / y), order=3)
test = torch.from_numpy(np.array([test.astype(np.float32)])).unsqueeze(0)