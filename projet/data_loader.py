import torch
import numpy as np
import os
import random
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from result_processing import write_to_file_pred

# definition of palettes from the get_palette.py, calculated with an annotation and classes
palette=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5], [0, 0, 6], [0, 0, 7]]
classes=['background', 'farmland', 'garden', 'woodland', 'grass', 'water', 'road', 'building']
root_img = r'/home/storage/zj/202303_08m/img_all_1024/'
root_ann = r'/home/storage/zj/202303_08m/mask_all_1024/'

colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8) # turn palette to indice
for i, colormap in enumerate(palette):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

def shuffle_images_labels(is_shuffle = True):
  images = []
  labels = []
  files_images = os.listdir(root_img) 
  if is_shuffle:
    random.shuffle(files_images)
    for file_image in files_images:
        a = os.path.join(root_img,file_image)
        images.append(a)
    for file_ann in files_images: # the order of annotation must be the same as the order of image after shuffling
        b = os.path.join(root_ann,file_ann)
        labels.append(b)
    print("Dataset shuffled")
    return images, labels
    
  if not is_shuffle: # if not shuffle, then the order is unchanged
    for file_image in files_images:
        a = os.path.join(root_img,file_image)
        images.append(a)
    files_anns = os.listdir(root_ann)
    for file_ann in files_anns:
        b = os.path.join(root_ann,file_ann)
        labels.append(b)
    print("Dataset not shuffled")
    return images, labels

#The dataset is divided into two groups, training and testing, 
#and a certain number of images are extracted specifically for prediction
def split_train_test_pred(images_shuffled, labels_shuffled, train_ratio, number, file_path):
    if number != 0:
      write_to_file_pred(file_path + "history.txt", images_shuffled[:number])
      images_shuffled = images_shuffled[number:]
      labels_shuffled = labels_shuffled[number:]
      
    total_samples = len(images_shuffled)
    train_samples = int(total_samples * train_ratio)
    images_train = images_shuffled[:train_samples]
    labels_train = labels_shuffled[:train_samples]
    images_test = images_shuffled[train_samples:]
    labels_test = labels_shuffled[train_samples:]
    print("Dataset splitted")
    print("Number for training: {}".format(len(images_train)))
    print("Number for test: {}".format(len(images_test)))
    print("Number for prediction: {}\n".format(number))
    return images_train, labels_train, images_test, labels_test

# A portion of data from the training dataset is selected for training
# and a portion of data from the test dataset is selected for testing 
# according to a certain ratio. This ratio can be customised.
def select_samples(images_train, labels_train, images_test, labels_test, sample_ratio):
    num_samples_train = int(len(images_train) * sample_ratio)
    images_train_final = images_train[:num_samples_train]
    labels_train_final = labels_train[:num_samples_train]
    num_samples_test = int(len(images_test) * sample_ratio)
    images_test_final = images_test[:num_samples_test]
    labels_test_final = labels_test[:num_samples_test]
    print('Training dataset:' + str(len(images_train_final)))
    print('Test dataset:' + str(len(images_test_final)))
    return images_train_final, labels_train_final, images_test_final, labels_test_final

# Functions dedicated to unet. 
def unet_resize_labels(labels, size):
    resized_labels = []
    for label in labels:
        resized_label = label.resize((size, size), Image.NEAREST)
        resized_labels.append(resized_label)
    return resized_labels

def label_indices(colormap): #turn label to indice
    colormap = np.array(colormap).astype('int32')
    return colormap2label[colormap]

class MyDataset(Dataset):
    def __init__(self, images, labels, is_train, is_augmented):

        self.rgb_mean = np.array([0.34, 0.379, 0.341]) #calculate by rgb_new.py
        self.rgb_std = np.array([0.16, 0.1533, 0.1335])

        #self.rgb_mean = np.array([86.795, 96.6289, 86.966])
        #self.rgb_std = np.array([40.713, 39.0967, 34.0515])
        self.tsf = transforms.Compose([ # Normalize the input images
            transforms.ToTensor(),
            transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)])
        #self.size = size
        self.is_augmented = is_augmented
        self.images = images
        self.labels = labels
        
        if is_train == True:      
          #self.images = self.filter(images)  
          #self.labels = self.filter(labels)  
          print('Read ' + str(len(self.images)) + ' valid examples for training')
        if is_train == False:
          #self.images = self.filter(images)  
          #self.labels = self.filter(labels)
          print('Read ' + str(len(self.images)) + ' valid examples for test')

    def filter(self, imgs): # Make sure the size of the input image is correct to prevent too small images in the dataset
        if self.is_augmented == True: 
            return [img for img in imgs if (
                img.size[1] >= self.size[0] and
                img.size[0] >= self.size[1])] 
        else:      
            return [img for img in imgs if (
                Image.open(img).size[1] >= self.size[0] and
                Image.open(img).size[0] >= self.size[1])]

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]
        if self.is_augmented == False:
            image = Image.open(image).convert('RGB')
            label = Image.open(label).convert('L')
        image = self.tsf(image)
        label = label_indices(label)

        return image, label  

    def __len__(self):
        return len(self.images)
