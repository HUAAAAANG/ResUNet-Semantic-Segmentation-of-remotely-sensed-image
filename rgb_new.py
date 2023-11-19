import os
import numpy as np
from PIL import Image

#define a function to calculate the mean and std of whole dataset
#these values will be used in transforms.Normalize function in data_loader
def calculate_channel_mean_std(image_folders):
    channel_means = [0, 0, 0]
    channel_stds = [0, 0, 0]
    count = 0
    
    for folder in image_folders:
        image_files = [f for f in os.listdir(folder) if f.endswith('.tif')] # get all image files
        
        for image_file in image_files:
            image_path = os.path.join(folder, image_file)
            image = Image.open(image_path)
            image_array = np.array(image)
            
            for i in range(3): 
                channel = image_array[:, :, i] #calculate mean and std value channel by channel for 1 image
                channel_means[i] += np.mean(channel)
                channel_stds[i] += np.std(channel)
            
            count += 1
    
    for i in range(3):
        channel_means[i] /= count # average value of whole dataset
        channel_stds[i] /= count
    
    return channel_means, channel_stds

folder1 = r'/home/storage/zj/202303_08m/img_all_1024/'
#folder1 = r'/home/test123/mmsegmentation/data/my_dataset/img_dir_train/'
#folder2 = r'/home/test123/mmsegmentation/data/my_dataset/img_dir_test/'
#folder3 = r'/home/test123/mmsegmentation/data/my_dataset/img_dir_validation/'

channel_means, channel_stds = calculate_channel_mean_std([folder1])

for i in range(3):
    print("Channel {}: Mean = {}, Std = {}".format(i, channel_means[i], channel_stds[i]))
