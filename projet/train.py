import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
from data_loader import MyDataset, shuffle_images_labels, split_train_test_pred, select_samples, unet_resize_labels
from data_augmentation import data_augmentation
from result_processing import calculate_metrics, print_matrix, plot_cm, plot_accuracy, plot_loss
from train_history import train_history

from models.fcn import FCN32s
from models.fcn_vgg import VGGNet, FCN8s
from models.unet import UNet
from models.unet_resnet import UResnet, BasicBlock, BottleNeck
from models.unet_resnet_pretrained import UResnet_pretrained
from models.resunet_a import ResUnet_a_d6
from models.resunet_a_d6_multitask import ResUnet_a_d6_Multitask

matplotlib.use('Agg')

# definition of classes, palettes
classes=['background', 'farmland', 'garden', 'woodland', 'grass', 'water', 'road', 'building']
palette=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5], [0, 0, 6], [0, 0, 7]]
classNum = 8

#choose your device
device = torch.device("cuda:1")

# fix the parameters
epochs = 1000
batchSize = 4
train_ratio = 0.8 # 80% for train, 20% for test
sample_ratio = 0.853 # 80% * 85.3% for train, 20% * 85.3% for test
#0.513
#0.853
#0.682
# totally 1176 images in the whole dataset
number_pred = 2 # 2 images for prediction visualization
file_path = "/home/test123/try/projet/result_resunet/test3/" # Select the path to save the training logs and model files

images, labels = shuffle_images_labels(is_shuffle = True)
images_train, labels_train, images_test, labels_test = split_train_test_pred(images, labels, train_ratio, number_pred, file_path) # random split
images_train, labels_train, images_test, labels_test = select_samples(images_train, labels_train, images_test, labels_test, sample_ratio) # random pick

images_train, labels_train = data_augmentation(images_train, labels_train, 
                                               is_hflip = False, hf_prob = 0.5, 
                                               is_vflip = False, vf_prob = 0.5, 
                                               is_noise = False, noise_prob = 0.2, dev = 0.1,
                                               is_blur = False, blur_prob = 0.2, kernel_size=(3, 3), sigma=0,
                                               is_box = False, box_prob = 0.2, box_size=32, number = 10,
                                               is_rotate = False, rot_prob = 0.2, degree_range = (0, 90),
                                               is_crop = True, crop_prob = 1.1, crop_size = (512, 512)) # random crop for sure

 # Same data augmentation operation on the test image
images_test, labels_test = data_augmentation(images_test, labels_test, is_crop = True, crop_prob = 1.1, crop_size = (512, 512))

#labels_train = unet_resize_labels(labels_train, 504)
#labels_test = unet_resize_labels(labels_test, 504)

# load data
dataLoaderTrain = MyDataset(images_train, labels_train, is_train=True, is_augmented = True)
dataLoaderTest = MyDataset(images_test, labels_test, is_train=False, is_augmented = True)
dataLoaderTrain = DataLoader(dataset=dataLoaderTrain, batch_size=batchSize, shuffle=True, num_workers=8, pin_memory=True)
dataLoaderTest = DataLoader(dataset=dataLoaderTest, batch_size=batchSize, shuffle=True, num_workers=8, pin_memory=True)

# select loss function
lossFn = nn.CrossEntropyLoss()

# select a model
#net = FCN32s(classNum)

#pretrain_model = VGGNet(requires_grad = True)
#net = FCN8s(pretrain_model, classNum)

#net = UNet(n_channels = 3, n_classes = classNum)

#net = UResnet(block = BasicBlock, layers = [3,4,6,3], num_classes = classNum)  # 34

#net = UResnet_pretrained('resnet34', classNum)

net = ResUnet_a_d6(3, classNum)

#net = ResUnet_a_d6_Multitask(3, classNum)

model_name = net.__class__.__name__

# select an optimizer
#optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-04, weight_decay=0.000005, amsgrad=False)
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)

# select a scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95, last_epoch=-1)

print("----- Net -----")
print(net)
print("")

# Obtain training and testing metrics and results
conf_matrix_train, conf_matrix_test, train_acc_list, test_acc_list, train_loss_list, test_loss_list = train_history(device, epochs, classNum, dataLoaderTrain, dataLoaderTest, net, lossFn, optimizer, scheduler, file_path)

# Visualisation of metrics and results in graphical form
#print_matrix(conf_matrix)
fig1 = plot_cm(conf_matrix_train, classNum, classes, epochs, file_path, is_train = True)
fig2 = plot_cm(conf_matrix_test, classNum, classes, epochs, file_path, is_train = False)
fig3 = plot_accuracy(train_acc_list, test_acc_list, epochs, file_path)
fig4 = plot_loss(train_loss_list, test_loss_list, epochs, file_path)
#plt.show()

#save the trained model
torch.save(net, file_path + f"{model_name}.pth")