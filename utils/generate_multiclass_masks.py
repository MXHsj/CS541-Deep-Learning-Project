# =======================================================================
# file name:    generate_multiclass_masks.py
# description:  generate masks for multi-class (pleural line & rib shadow)
#               smantic segmentation from raw binary masks
# authors:      Xihan Ma, Mingjie Zeng, Xiaofan Zhou
# date:         2022-11-13
# version:
# =======================================================================
import os
import cv2
import numpy as np

# TODO: use this script to process phantom data

IMG_PATH = os.path.join(os.path.dirname(__file__), '../dataset_patient/image')
PLEURAL_PATH = os.path.join(os.path.dirname(__file__), '../dataset_patient/masks/pleural_line')
# RIB_LINE_PATH = os.path.join(os.path.dirname(__file__), '../dataset_patient/masks/rib_line')
RIB_SHADOW_PATH = os.path.join(os.path.dirname(__file__), '../dataset_patient/masks/rib_shadow')
MASK_PATH = os.path.join(os.path.dirname(__file__), '../dataset_patient/mask_merged')

img_path_list = os.listdir(IMG_PATH)  # bmode image
pleural_path_list = os.listdir(PLEURAL_PATH)  # mask - pleural line
# rib_line_path_list = os.listdir(RIB_LINE_PATH) # mask - rib line
rib_shadow_path_list = os.listdir(RIB_SHADOW_PATH)  # mask - rib shadow

for img_path in img_path_list:
  img_name = img_path.split('.')[0]
  # print(img_name)
  img = cv2.imread(os.path.join(IMG_PATH, img_path), cv2.IMREAD_GRAYSCALE)
  h, w = img.shape[:2]
  #print(f'h:{h}, w:{w}')

  pl_msk_path = img_name + '-pleural_line.jpg'
  rs_msk_path = img_name + '-rib_shadow.jpg'

  if pleural_path_list.count(pl_msk_path) == 1:
    pl = cv2.imread(os.path.join(PLEURAL_PATH, pl_msk_path), cv2.IMREAD_GRAYSCALE)
  else:
    pl = np.zeros([h, w, 1], np.uint8)  # if no mask file found, generate all 0 mask

  if rib_shadow_path_list.count(rs_msk_path) == 1:
    rs = cv2.imread(os.path.join(RIB_SHADOW_PATH, rs_msk_path), cv2.IMREAD_GRAYSCALE)
  else:
    rs = np.zeros([h, w, 1], np.uint8)  # if no mask file found, generate all 0 mask

  merge = np.zeros([h, w, 1], np.uint8)   # background ==> 0
  merge[pl[:] > 200] = 1                  # pleural line ==> 1
  merge[rs[:] > 200] = 2                  # rib shadow ==> 2

  path2write = MASK_PATH + '/' + img_name + '_msk.jpg'
  cv2.imwrite(path2write, merge)
  print(f'new mask saved: {path2write}')
