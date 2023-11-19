from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

filep = "/home/storage/zj/202303_08m/mask_all_1024/2_6_H51G007006.tif"                 
mask_filename = os.path.basename(filep)
mask_name, mask_ext = os.path.splitext(mask_filename)

# Recolouring the annotation for better visualization                             
img = Image.open(filep)  
pic = img.convert('RGB')           
print(pic.size[0],pic.size[1])
for x in range(pic.size[0]):
            for y in range(pic.size[1]):
                r,g,b = pic.getpixel((x,y)) #Here, due to the function used, the value in the R channel is the category
                if r == 0:
                   pic.putpixel((x,y),(255, 255, 255))
                if r == 1:
                   pic.putpixel((x,y),(255, 0, 0))
                if r == 2:
                   pic.putpixel((x,y),(0, 255, 0))
                if r == 3:
                   pic.putpixel((x,y),(0, 0, 255))
                if r == 4:
                   pic.putpixel((x,y),(255, 255, 0))
                if r == 5:
                   pic.putpixel((x,y),(255, 0, 255))
                if r == 6:
                   pic.putpixel((x,y),(0, 255, 255))
                if r == 7:
                   pic.putpixel((x,y),(0, 0, 0))

#Save the recolored annotation
pic.save("/home/test123/try/projet/result/rgb_" + mask_name + ".jpg")