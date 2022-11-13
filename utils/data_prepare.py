import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

sample_list = []
for i in os.listdir('1'):
    sample_list.append('1/' + i)
for i in os.listdir('2'):
    sample_list.append('2/' + i)
for sample in sample_list:
    pleural_line = 'Labels/' + sample.split('.')[0] + '-pleural_line.jpg'
    rib_line = 'Labels/' + sample.split('.')[0] + '-rib_line.jpg'
    rib_shadow = 'Labels/' + sample.split('.')[0] + '-rib_shadow.jpg'
    pl = cv2.imread(pleural_line, cv2.IMREAD_GRAYSCALE)
    rl = cv2.imread(rib_line, cv2.IMREAD_GRAYSCALE)
    rs = cv2.imread(rib_shadow, cv2.IMREAD_GRAYSCALE)
    print(sample)
    merge = pl / 255 + 2 * rl / 255 + 3 * rs / 255
    # merge = np.ones([pl.shape[0], pl.shape[1], 1])
    # merge[:, :, 0] = pl
    # merge[:, :, 1] = rl
    # merge[:, :, 2] = rs
    # merge=merge
    cv2.imwrite('mask/' + sample, merge)
