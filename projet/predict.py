import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import os

#choose your device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#definition of recolored palette and classes
palette_predict=[[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 0, 0]]
classes=['background', 'farmland', 'garden', 'woodland', 'grass', 'water', 'road', 'building']

#Select a network model that has been trained to make predictions
model = torch.load("/home/test123/try/projet/result_unet/unet/test3/UNet_1000.pth")

# use the same mean and std as the whole dataset to normalize the image
#mean and std calculated by rgb_new.py
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.34, 0.379, 0.341], std=[0.16, 0.1533, 0.1335])
])

# load the image for prediction
image_path = "/home/storage/zj/202303_08m/img_all_1024/2_6_H51G007006.tif"
image_filename = os.path.basename(image_path)
image_name, image_ext = os.path.splitext(image_filename)

image = Image.open(image_path)
image = preprocess(image).unsqueeze(0)

#make a prediction using the model
model.to(device)
image = image.to(device)

#do not modify the weight of model
model.eval()

with torch.no_grad():
   output = model(image)
 
output = output.squeeze(0)
output = torch.softmax(output, dim=0) # probability of classes
output = output.argmax(dim=0) # prediction result
pred = output.cpu().numpy()

# The following section reconstructs the image based on the prediction results, 
# i.e. the category corresponding to each pixel point, and compares it with the recoloured annotation.
palette = np.array(palette_predict, dtype = np.uint8)

colored_pred = palette[pred]

img_pred = Image.fromarray(colored_pred, mode = 'RGB')

label_width = 200 
label_height = img_pred.height/8  

result_width = img_pred.width + label_width
result_height = img_pred.height
result_image = Image.new('RGB', (result_width, result_height))

result_image.paste(img_pred, (0, 0))
draw = ImageDraw.Draw(result_image)
label_x = img_pred.width 

font_size = 25
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Oblique.ttf", font_size)

for i, color in enumerate(palette_predict):
    label_y = i * label_height  
    draw.rectangle([(label_x, label_y), (label_x + 50, label_y + label_height)], fill=tuple(color))
    draw.text((label_x + 60, label_y + label_height/8 + 40), str(classes[i]), font=font, fill=(255, 255, 255))

output_path = "/home/test123/try/projet/result/prediction_" + image_name + ".jpg"
result_image.save(output_path)