import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_Lung, trainer_encoder
import cv2
from scipy.ndimage.interpolation import zoom
from PIL import Image
from torchvision import transforms
from datasets.dataset_synapse import Lung_dataset, RandomGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../dataset/train',
                    help='root dir for data')
parser.add_argument('--encoder_path', type=str,
                    default='../dataset_patient_nolabel',
                    help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Lung', help='experiment_name')

parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--encoder_max_epochs', type=int,
                    default=10, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--encoder', type=bool,
                    default=False, help='vit_patches_size, default is 16')
args = parser.parse_args()


def showing_seg(img_path, pred_arr):
    img_arr = cv2.imread(img_path)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for j in range(1, pred_arr.shape[0]):
        edge_arr = cv2.dilate(cv2.Canny(pred_arr[j, :, :], 50, 150), kernel, iterations=1)
        edge_arr = cv2.cvtColor(edge_arr, cv2.COLOR_GRAY2BGR)
        img_arr *= (1 - edge_arr // 255)
        if j // 4 == 0:
            edge_arr[:, :, 0] = 0
        if j % 4 < 2:
            edge_arr[:, :, 1] = 0
        if j % 2 == 0:
            edge_arr[:, :, -1] = 0
        img_arr += edge_arr
    return img_arr  # [H, W, 3]


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Lung': {
            'root_path': '../dataset_patient',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 4,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.is_pretrain = False
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]

    config_vit.n_skip = args.n_skip
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = {'Lung': trainer_Lung, 'encoder': trainer_encoder}
    if args.encoder == True:
        encoder_path = "../model/{}/{}".format(args.exp, 'encoder')
        if False:
            config_vit.n_classes = 1
            net = ViT_seg(config_vit, img_size=args.img_size, num_classes=1).to(device=device)
            
            if not os.path.exists(encoder_path):
                os.makedirs(encoder_path)
            trainer['encoder'](args, net, encoder_path)
    config_vit.n_classes = args.num_classes
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device=device)
    if args.encoder == True:
        weights = torch.load(encoder_path+'/epoch_9.pth')
        net.load_encoder(weights=weights)
    else:
        weights = np.load(config_vit.pretrained_path)
        net.load_from(weights=weights)
    trainer[dataset_name](args, net, snapshot_path)
