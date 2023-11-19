import os
import torch
import cv2
import numpy as np

data_dir = r'/home/test123/mmsegmentation/data/my_dataset/img_dir_train/'
print('Number:',len(os.listdir(data_dir)))

batch_size = len(os.listdir(data_dir))
h,w = 256, 256 
batch = torch.zeros(batch_size, 3, h, w, dtype=torch.uint8)

filenames = [name for name in os.listdir(data_dir)]
for i, filename in enumerate(filenames): 
    img_arr = cv2.imread(os.path.join(data_dir, filename))
    assert img_arr is not None, 'Failed' 
    img_arr = np.resize(img_arr,(h,w,3))
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2,0,1) 
    img_t = img_t[:3] 
    batch[i] = img_t

batch = batch.float()
batch /= 255.
n_channels = batch.shape[1]
means, stds = [], []
for c in range(n_channels):
    mean = torch.mean(batch[:,c])
    means.append(mean)
    std = torch.std(batch[:,c])
    stds.append(std)
    batch[:,c] = (batch[:,c] - mean) / std
print("RGB Mean: ", means)
print("RGB Std: ", stds)