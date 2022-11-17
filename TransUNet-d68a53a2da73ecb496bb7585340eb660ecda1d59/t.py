import cv2
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path))