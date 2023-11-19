from PIL import Image

# choose an annotation that has all the categories on it.
image = Image.open("/home/storage/zj/202303_08m/mask_all_1024/2_4_H51G002007.tif")

width, height = image.size

r_values = set()
g_values = set()
b_values = set()

#get the rgb value of annotation by using right shift
for y in range(height):
    for x in range(width):
        pixel_value = image.getpixel((x, y))
        r = (pixel_value >> 16) & 0xFF  
        g = (pixel_value >> 8) & 0xFF 
        b = pixel_value & 0xFF  
        r_values.add(r)
        g_values.add(g)
        b_values.add(b)
      
#test the output rgb values, normally r=0, g=0, b=[0,7]
print("r:")
for r in r_values:
    print(r)
    
print("g:")
for g in g_values:
    print(g)
    
print("b:")
for b in b_values:
     print(b)