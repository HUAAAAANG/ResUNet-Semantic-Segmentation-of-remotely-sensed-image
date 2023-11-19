import torch
import torch.nn as nn
import torch.nn.functional as F

# This model is very similar to ResUnet-a-d6
# Except the output layer
class Conv2DN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.normed_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        
    def forward(self, x):
        return self.normed_conv(x)

class ResBlock_4(nn.Module):
    def __init__(self, channel, kernel_size, dilation, stride):
        super().__init__()
        
        self.d1_4 = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=dilation[0]*(kernel_size-1)//2, dilation=dilation[0], bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=dilation[0]*(kernel_size-1)//2, dilation=dilation[0], bias=True)
        )
        
        self.d2_4 = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=dilation[1]*(kernel_size-1)//2, dilation=dilation[1], bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=dilation[1]*(kernel_size-1)//2, dilation=dilation[1], bias=True)
        )
        
        self.d3_4 = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=dilation[2]*(kernel_size-1)//2, dilation=dilation[2], bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=dilation[2]*(kernel_size-1)//2, dilation=dilation[2], bias=True)
        )
        
        self.d4_4 = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=dilation[3]*(kernel_size-1)//2, dilation=dilation[3], bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=dilation[3]*(kernel_size-1)//2, dilation=dilation[3], bias=True)
        )
    
    def forward(self, x):
        return self.d1_4(x) + self.d2_4(x) + self.d3_4(x) +self.d4_4(x) + x
    
class ResBlock_3(nn.Module):
    def __init__(self, channel, kernel_size, dilation, stride):
        super().__init__()
        
        self.d1_3 = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=dilation[0]*(kernel_size-1)//2, dilation=dilation[0], bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=dilation[0]*(kernel_size-1)//2, dilation=dilation[0], bias=True)
        )
        
        self.d2_3 = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=dilation[1]*(kernel_size-1)//2, dilation=dilation[1], bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=dilation[1]*(kernel_size-1)//2, dilation=dilation[1], bias=True)
        )
        
        self.d3_3 = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=dilation[2]*(kernel_size-1)//2, dilation=dilation[2], bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=dilation[2]*(kernel_size-1)//2, dilation=dilation[2], bias=True)
        )
    
    def forward(self, x):
        return self.d1_3(x) + self.d2_3(x) + self.d3_3(x) + x
    
class ResBlock_1(nn.Module):
    def __init__(self, channel, kernel_size, dilation, stride):
        super().__init__()
        
        self.d1_1 = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=dilation * (3 - 1)//2, dilation=dilation, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=dilation * (3 - 1)//2, dilation=dilation, bias=True)
        )
    
    def forward(self, x):
        return self.d1_1(x) + x
    
class PSP_Pooling(nn.Module):
    def __init__(self, channel):
        super().__init__()
        
        self.pooling_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=1),
            Conv2DN(channel//4,channel//4)
        )
        
        self.pooling_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            Conv2DN(channel//4,channel//4)
        )
        
        self.pooling_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            Conv2DN(channel//4,channel//4)
        )
        
        self.pooling_8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8),
            Conv2DN(channel//4,channel//4)
        )
        
        self.conv2dn_cat = Conv2DN(channel*2, channel)

    def split(self,x):
        channels = x.size(1)  
        split_size = channels // 4 

        x_1 = x[:, :split_size, :, :]
        x_2 = x[:, split_size:2*split_size, :, :]
        x_3 = x[:, 2*split_size:3*split_size, :, :]
        x_4 = x[:, 3*split_size:, :, :]

        return x_1, x_2, x_3, x_4
          
    def forward(self, x):
        x_1, x_2, x_3, x_4 = self.split(x)
        h, w=x.size(2), x.size(3)

        x = [F.interpolate(self.pooling_1(x_1), size=(h,w), mode='nearest'),
             F.interpolate(self.pooling_2(x_2), size=(h,w), mode='nearest'),
             F.interpolate(self.pooling_4(x_3), size=(h,w), mode='nearest'),
             F.interpolate(self.pooling_8(x_4), size=(h,w), mode='nearest'),
             F.interpolate(x, size=(h,w), mode='nearest')]
             
        x = torch.cat(x, dim=1)
        x = self.conv2dn_cat(x)
        return x

class Upsampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.conv2dn_up = Conv2DN(in_channel, out_channel)
        
    def upsample(self, x, scale_factor):
        return F.interpolate(x, scale_factor=scale_factor, mode='nearest')
        
    def forward(self, x):
        x = self.upsample(x, scale_factor=2)
        x = self.conv2dn_up(x)
        return x
        
class Combine(nn.Module):
    def __init__(self, channel):
        super().__init__()
        
        self.relu = nn.ReLU()
        self.conv2dn_combine = Conv2DN(channel*2, channel)
        
    def forward(self, x1, x2):
        x1 = self.relu(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2dn_combine(x)
        return x

# There are 4 outputs in total

# The output of Color Task layer is the colour logic
class Color_Task(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.color = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)
        
    def forward(self, x):
        x = self.color(x)
        return x
    
# The output of Distance Task layer is the distance logic
class Distance_Task(nn.Module):
    def __init__(self, in_channel, middle_channel, out_channel):
        super().__init__()
        
        self.distance = nn.Sequential(
            nn.Conv2d(in_channel, middle_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(),
            nn.Conv2d(middle_channel, middle_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(),
            nn.Conv2d(middle_channel, out_channel, kernel_size=1, padding=0)
        )
        
    def forward(self, x):
        x = self.distance(x)
        return x

# The output of Bound Task layer is the bound logic
class Bound_Task(nn.Module):
    def __init__(self, in_channel, middle_channel, out_channel):
        super().__init__()
        
        self.bound = nn.Sequential(
            nn.Conv2d(in_channel, middle_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(),
            nn.Conv2d(middle_channel, out_channel, kernel_size=1, padding=0)
        )
        
    def forward(self, x):
        x = self.bound(x)
        return(x)
  
# The output of Segmentation Task layer is the segmentation logic      
class Segmentation_Task(nn.Module):
    def __init__(self, in_channel, middle_channel, out_channel):
        super().__init__()
        
        self.seg = nn.Sequential(
            nn.Conv2d(in_channel, middle_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(),
            nn.Conv2d(middle_channel, middle_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(),
            nn.Conv2d(middle_channel, out_channel, kernel_size=1, padding=0)
        )
        
    def forward(self, x):
        x = self.seg(x)
        return x

class ResUnet_a_d6_Multitask(nn.Module):
    def __init__(self, in_channel, n_class):
        super().__init__()
        
        self.in_channel = in_channel
        self.n_class = n_class
        
        self.layer1 = nn.Conv2d(self.in_channel, 32, kernel_size=1, dilation=1, stride=1)
        self.layer2 = ResBlock_4(32, 3, [1,3,15,31], 1)
        self.layer3 = nn.Conv2d(32, 64, kernel_size=1, dilation=1, stride=2)
        self.layer4 = ResBlock_4(64, 3, [1,3,15,31], 1)
        self.layer5 = nn.Conv2d(64, 128, kernel_size=1, dilation=1, stride=2)
        self.layer6 = ResBlock_3(128, 3, [1,3,15], 1)
        self.layer7 = nn.Conv2d(128, 256, kernel_size=1, dilation=1, stride=2)
        self.layer8 = ResBlock_3(256, 3, [1,3,15], 1)
        self.layer9 = nn.Conv2d(256, 512, kernel_size=1, dilation=1, stride=2)
        self.layer10 = ResBlock_1(512, 3, 1, 1)
        self.layer11 = nn.Conv2d(512, 1024, kernel_size=1, dilation=1, stride=2)
        self.layer12 = ResBlock_1(1024, 3, 1, 1)
        self.layer13 = PSP_Pooling(1024)
        self.layer14 = Upsampling(1024,512)
        self.layer15 = Combine(512)
        self.layer16 = ResBlock_1(512, 3, 1, 1)
        self.layer17 = Upsampling(512,256)
        self.layer18 = Combine(256)
        self.layer19 = ResBlock_1(256, 3, 1, 1)
        self.layer20 = Upsampling(256,128)
        self.layer21 = Combine(128)
        self.layer22 = ResBlock_1(128, 3, 1, 1)
        self.layer23 = Upsampling(128, 64)
        self.layer24 = Combine(64)
        self.layer25 = ResBlock_1(64, 3, 1, 1)
        self.layer26 = Upsampling(64,32)
        self.layer27 = Combine(32)
        self.layer28 = ResBlock_1(32, 3, 1, 1)
        self.layer_last_combine = Combine(32)
        self.layer_last_pooling = PSP_Pooling(32)
        self.layer_last_relu = nn.ReLU()
    
# Conditional multitask output layer
        self.layer_color = Color_Task(32,3)
        self.layer_distance = Distance_Task(32, 32, self.n_class)
        self.layer_bound = Bound_Task(32+self.n_class, 32, self.n_class)
        self.layer_seg = Segmentation_Task(3+self.n_class*2, 32, self.n_class)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x1=self.layer1(x)
        x2=self.layer2(x1)
        x3=self.layer3(x2)
        x4=self.layer4(x3)
        x5=self.layer5(x4)
        x6=self.layer6(x5)
        x7=self.layer7(x6)
        x8=self.layer8(x7)
        x9=self.layer9(x8)
        x10=self.layer10(x9)
        x11=self.layer11(x10)
        x12=self.layer12(x11)
        x13=self.layer13(x12)
        x14=self.layer14(x13)
        x15=self.layer15(x14, x10)
        x16=self.layer16(x15)
        x17=self.layer17(x16)
        x18=self.layer18(x17, x8)
        x19=self.layer19(x18)
        x20=self.layer20(x19)
        x21=self.layer21(x20, x6)
        x22=self.layer22(x21)
        x23=self.layer23(x22)
        x24=self.layer24(x23, x4)
        x25=self.layer25(x24)
        x26=self.layer26(x25)
        x27=self.layer27(x26, x2)
        x28=self.layer28(x27)
        x_lc=self.layer_last_combine(x28, x1)
        x_lp=self.layer_last_pooling(x_lc)
        x_relu=self.layer_last_relu(x_lp)
        
        x_color=self.layer_color(x_lc)
        x_color_seg=self.sigmoid(x_color) # color outpput 
        
        x_dis=self.layer_distance(x_lc)
        x_dis_bound=self.softmax(x_dis)
        x_dis_seg=x_dis_bound # distance output
        
        x_bound=torch.cat([x_relu, x_dis_bound], dim=1)
        x_bound=self.layer_bound(x_bound)
        x_bound_seg=self.sigmoid(x_bound) # bound output
        
        x_seg=torch.cat([x_color_seg, x_bound_seg, x_dis_seg], dim=1)
        x_seg=self.layer_seg(x_seg) # segmentation output
        
        return x_seg, x_bound, x_dis, x_color
# Output shape is : [1, 8, 512, 512], [1, 8, 512, 512], [1, 8, 512, 512], [1, 3, 512, 512]
# 8 for classes, 3 for rgb

#image = torch.randn(1, 3, 512, 512) 
#model = ResUnet_a_d6_Multitask(3, 8)
#x_seg, x_bound, x_dis, x_color = model(image)
#print("Output shape:", x_seg.shape, x_bound.shape, x_dis.shape, x_color.shape)