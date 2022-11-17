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
from PIL import Image
from torchvision import transforms
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/Users/xfdw/Desktop/wpi term1/deeplearning/project/input/LUS_patient_baseline/dataset',
                    help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='LUS_patient_baseline', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
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
    # if not args.deterministic:
    #    cudnn.benchmark = True
    #    cudnn.deterministic = False
    # else:
    #    cudnn.benchmark = False
    #    cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'LUS_patient_baseline': {
            'root_path': '../input/LUS_patient_baseline/dataset',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 3,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device=device)
    # weights = np.load(config_vit.pretrained_path)
    # net.load_from(weights=weights)
    weights = torch.load('../model/TU_LUS_patient_baseline224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224'
                         '/epoch_149.pth')
    net.load_state_dict(weights)
    test = cv2.imread('../input/LUS_patient_baseline/dataset/2/man-03-L-2-V-frame1.jpg', cv2.IMREAD_GRAYSCALE)
    x, y = test.shape
    test = zoom(test, (224 / x, 224 / y), order=3)
    test = torch.from_numpy(np.array([test.astype(np.float32)])).unsqueeze(0)

    a = net(test)
    out = torch.argmax(torch.softmax(a, dim=1), dim=1).squeeze(0)
    out = 255*out.cpu().detach().numpy()
    pred = zoom(out, (x / 1124, y / 820), order=0)
    #prediction[ind] = pred
    cv2.imwrite('test.png',pred)
    # trainer = {'LUS_patient_baseline': trainer_synapse,}
    # trainer[dataset_name](args, net, snapshot_path)
