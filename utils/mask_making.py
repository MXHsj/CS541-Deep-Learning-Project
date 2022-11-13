import os
import cv2
import numpy as np

# multi classes
class_types = ['background', 'pleural_line', 'rib_shadow']

IMG_PATH = 'dataset_patient_new/image'
PLEURAL_PATH = 'dataset_patient_new/masks/pleural_line'
#RIB_LINE_PATH = 'dataset_patient_new/masks/rib_line'
RIB_SHADOW_PATH = 'dataset_patient_new/masks/rib_shadow'

MASK_PATH = 'dataset_patient_new/mask_new'

# image
img_path_list = os.listdir(IMG_PATH)

# mask - pleural line
pleural_path_list = os.listdir(PLEURAL_PATH)

# mask - rib line
#rib_line_path_list = os.listdir(RIB_LINE_PATH)

# mask - rib shadow
rib_shadow_path_list = os.listdir(RIB_SHADOW_PATH)


for img_path in img_path_list:
    img_name = img_path.split('.')[0]
    target_length = len(img_name)
    print(img_path)
    img = cv2.imread(os.path.join(IMG_PATH, img_path), cv2.IMREAD_GRAYSCALE)
    print(f"shape: {img.shape}")

    h, w = img.shape[:2]
    #print(f"h:{h}, w:{w}")

    # empty mask
    #mask = np.zeros([h, w, 1], np.uint8)

    # black=0, white=1
    pl_count = 0
    rs_count = 0

    for pleural_path in pleural_path_list:
        pleural_name = pleural_path[:target_length]
        #print("test")
        #print(pleural_name)
        if pleural_name == img_name:
            pl_count += 1
            #print(f"rs: {PLEURAL_PATH + pleural_path}")
            pl = cv2.imread(os.path.join(PLEURAL_PATH, pleural_path), cv2.IMREAD_GRAYSCALE)
            #print(f"pl shape: {pl.shape}")

            for rib_shadow_path in rib_shadow_path_list:
                rib_shadow_name = rib_shadow_path[:target_length]
        
                if rib_shadow_name == img_name:
                    rs_count += 1
                    #print(f"rs: {RIB_SHADOW_PATH + pleural_path}")
                    rs = cv2.imread(os.path.join(RIB_SHADOW_PATH, rib_shadow_path), cv2.IMREAD_GRAYSCALE)
                    #print(f"rs shape: {rs.shape}")
                    break
            break
     
    #for rib_line_path in rib_line_path_list:
    #    rib_line_name = rib_line_path[:target_length]
    #    
    #    if rib_line_name == img_name:
    #        rl = cv2.imread(os.path.join(RIB_LINE_PATH, pleural_path), cv2.IMREAD_GRAYSCALE)
    #        #print(f"shape: {pl.shape}")
    #        break

    if pl_count == 0:
        pl = np.zeros([h, w], np.uint8)
    if rs_count == 0:
        rs = np.zeros([h, w], np.uint8)

    merge = pl / 255 + 2 * rs / 255

    cv2.imwrite(MASK_PATH + '/' + img_path, merge)

    
