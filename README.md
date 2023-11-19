# ResUNet-Semantic-Segmentation-of-remotely-sensed-image

In this project, the goal is to train a convolutional neural network with high accuracy for semantic segmentation of satellite remote sensing images. More specifically, satellite remote sensing imagery is classified into eight classes based on their land use categories: water, roads, buildings, agricultural arable land, grassland, gardens, woodland and background.
There are 1176 images in this dataset, and the size of each image is 1024 pixels high and 1024 pixels wide. The number of channels of the images is 3. The masks have been greyscaled beforehand and their pixel values are then equal to the ordinal number of the corresponding category.

In this project, I built a total of six different types of networks: full convolutional neural networks (FCN8s), full convolutional neural networks with VGG as a pre-trained model (FCN32s), UNET, ResUNet-34, ResUNet-a and Res-UNet-a-d6 with conditional multitask output layer.

In this repository:
1. get_palette.py contains a way to test the value of RGB in an annotation mask. After greyscaling, the B channel of the annotation mask has the value of the order number corresponding to the category, and the rest of the channels have a value of zero.
2. rgb_new.py contains a function that calculates the mean and standard deviation of the images of the entire dataset.
3. In the projet folder, data_augmentation.py contains methods for data augmentation. It is recommended that the chance of random cropping be constant at 1 and that the cropping size be 512*512.
4. data_loader.py is for loading data. It also provides for shuffling the data and dividing the database into training and testing parts.
5. parameter_counter.py contains the function that can be used to calculate the total number of parameters in a model, as well as the total number of parameters that can be trained.
6. predict.py provides the functionality to use the trained model for result prediction.
7. result_processing.py contains functions to calculate various training metrics.
8. show.py is used to visualise an annotation mask or the prediction result of neural network.
9. train.py is the main script for network training. It contains various tunable parameters and hyperparameters.
10. train_history.py documents how a network is trained and specifies how the training log is to be recorded.
11. In the model folder, here are seven network models. It is worth noting that for ResUNet-a-d6 with conditional multitask output layer, the corresponding training script also needs to be changed. This is because it has four outputs instead of one, which are collectively used for the computation of loss function and the updating of network weights, and because one of the outputs, the colour logic output, has a channel count of 3 unlike the other logic outputs which have a channel count of 8.

Due to confidentiality issues, the remote sensing satellite images I use are not allowed to be published. However, remote sensing satellite images on ImageNet are usable for my network. Change the definition of palette and class before use.
