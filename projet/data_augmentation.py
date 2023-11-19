import random
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import math

# Create a number of highly customisable data augmentation methods including cropping, rotation, 
# horizontal flip, vertical flip, Gaussian noise, Gaussian blur, black square and erasing

def data_augmentation(images, labels, is_crop = False, crop_prob = 0.5, crop_size = (512, 512),
                                      is_hflip = False, hf_prob = 0.5, 
                                      is_vflip = False, vf_prob = 0.5, 
                                      is_noise = False, noise_prob = 0.2, dev = 0.1,
                                      is_blur = False, blur_prob = 0.2, kernel_size=(3, 3), sigma=0,
                                      is_box = False, box_prob = 0.2, box_size=32, number = 10,
                                      is_rotate = False, rot_prob = 0.8, degree_range = (0, 90)):

    
#   imgs_aug = []
#   lbls_aug = []
   
#   for img, lbl in zip(images, labels):
#        img = Image.open(img).convert('RGB')
#        lbl = Image.open(lbl).convert('RGB')
#        imgs_aug.append(img)
#        lbls_aug.append(lbl)

# Enable different data augmentation methods as appropriate                                  
   if is_crop:
      images, labels = random_crop(images, labels, crop_prob, crop_size)
   if is_hflip:
      images, labels = random_horizontal_flip(images, labels, hf_prob)
   if is_vflip:
      images, labels = random_vertical_flip(images, labels, vf_prob)
   if is_noise:
      images, labels = add_gaussian_noise(images, noise_prob, dev)
   if is_blur:
      images, labels = add_gaussian_blur(images, blur_prob, kernel_size, sigma)
   if is_box:
      images, labels = add_square_blackboxes(images, labels, box_prob, box_size, number)
   if is_rotate:
      images, labels = random_rotation(images, labels, rot_prob, degree_range)

      
   return images, labels

# The probabilities of all the following methods can be customised
def random_horizontal_flip(images, labels, probability):
    flipped_images = []
    flipped_labels = []
    for img, lbl in zip(images, labels):
        if random.random() < probability:
            img = transforms.functional.hflip(img)
            lbl = transforms.functional.hflip(lbl)
        flipped_images.append(img)
        flipped_labels.append(lbl)
    return flipped_images, flipped_labels
    
def random_vertical_flip(images, labels, probability):
    flipped_images = []
    flipped_labels = []
    for img, lbl in zip(images, labels):   
        if random.random() < probability:
            img = transforms.functional.vflip(img)
            lbl = transforms.functional.vflip(lbl)
        flipped_images.append(img)
        flipped_labels.append(lbl)
    return flipped_images, flipped_labels
    
def add_gaussian_noise(images, probability, std_dev):
    noisy_images = []
    labels = []
    for img, lbl in zip(images, labels):
        if random.random() < probability:
            noise = torch.randn_like(img) * std_dev # add Gaussien noise for image
            noisy_img = img + noise
        noisy_images.append(noisy_img)
        labels.append(lbl)
    return noisy_images, labels
    
def add_gaussian_blur(images, probability, kernel_size, sigma):
    blurred_images = []
    labels = []
    for img, lbl in zip(images, labels):
        if random.random() < probability:
            blurred_img = cv2.GaussianBlur(img, kernel_size, sigma) # add Gaussien blur for image
        blurred_images.append(blurred_img)
        labels.append(lbl)
    return blurred_images, labels
    
def add_square_blackboxes(images, labels, probability, box_size, num_box):
    patched_images = []
    patched_labels = []
    for img, lbl in zip(images, labels):
        height, width = img.size(1), img.size(2)
        if random.random() < probability:
            for _ in range(num_box):
                x = random.randint(0, width - box_size)
                y = random.randint(0, height - box_size)
                img[:, y:y+box_size, x:x+box_size] = 0  # Add black patch
                lbl[:, y:y+box_size, x:x+box_size] = 255  # Set label to maximum value
        patched_images.append(img)
        patched_labels.append(lbl)
    return patched_images, patched_labels

def random_rotation(images, labels, probability, angle_range):
    rotated_images = []
    rotated_labels = []
    for img, lbl in zip(images, labels):
        if random.random() < probability:
            angle = random.uniform(*angle_range) # The angle of rotation is a range
            img = transforms.functional.rotate(img, angle)
            lbl = transforms.functional.rotate(lbl, angle)
        rotated_images.append(img)
        rotated_labels.append(lbl)
    return rotated_images, rotated_labels

#In this project, the possibility of random cropping is set to 1
def random_crop(images, labels, probability, size):
    cropped_images = []
    cropped_labels = []
    for img, lbl in zip(images, labels):
        img = Image.open(img).convert('RGB')
        lbl = Image.open(lbl).convert('L')
        if random.random() < probability:
            i, j, h, w = transforms.RandomCrop.get_params(img, size)
            img = transforms.functional.crop(img, i, j, h, w)
            lbl = transforms.functional.crop(lbl, i, j, h, w)
        cropped_images.append(img)
        cropped_labels.append(lbl)
    return cropped_images, cropped_labels

# Will not be used in this project, but a worthy try!
def random_erasing(images, labels, probability=0.5, sl=0.02, sh=0.4, r1=0.3, r2=1/0.3):
    erased_images = []
    erased_labels = []
    for img, lbl in zip(images, labels):
        if random.random() < probability:
            img = F.to_tensor(img)
            lbl = F.to_tensor(lbl)
            h, w = img.shape[1:]
            area = h * w

            for attempt in range(100):
                target_area = random.uniform(sl, sh) * area
                aspect_ratio = random.uniform(r1, r2)

                h_target = int(round(math.sqrt(target_area * aspect_ratio)))
                w_target = int(round(math.sqrt(target_area / aspect_ratio)))

                if w_target < w and h_target < h:
                    x1 = random.randint(0, w - w_target)
                    y1 = random.randint(0, h - h_target)
                    img[:, y1:y1+h_target, x1:x1+w_target] = 0
                    lbl[:, y1:y1+h_target, x1:x1+w_target] = 255
                    break
            
            img = F.to_pil_image(img)
            lbl = F.to_pil_image(lbl)
        
        erased_images.append(img)
        erased_labels.append(lbl)
    
    return erased_images, erased_labels