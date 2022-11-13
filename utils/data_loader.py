import os
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize

DATA_PATH = '../dataset_patinet/'
IMG_HEIGHT = 820
IMG_WIDTH = 1124
IMG_CHANNELS = 3

print(os.path(DATA_PATH))
train_ids = next(os.walk(DATA_PATH))[1]
print(train_ids)

# for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

#   path = DATA_PATH + "/" + id_
#   h, w = IMG_HEIGHT, IMG_WIDTH
#   img = imread(path + '/image/' + 'Frame' + id_ + '.jpg')[:, :, :IMG_CHANNELS]

#   # Image resizing to lower resolution
#   img = resize(img, (h, w), mode='constant', preserve_range=True)

#   mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
#   # print(path + '/mask/light/')
#   for _, _, mask_file in os.walk(path + '/mask/light/'):
#     mask_ = imread(path + '/mask/light/' + mask_file[0])

#     mask_ = resize(mask_, (h, w, 1), mode='constant', preserve_range=True)
#     mask = np.maximum(mask, mask_)
