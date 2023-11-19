import torch
import torch.nn as nn

# there are two type of basic bloack for ResNet
# If the model depth is less than 50 layers, the BasicBlock is used
class BasicBlock(nn.Module):          
    expansion = 1 # Channel Expansion Ratio
 
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # residual connection
        self.residual_forward = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        
        # shortcut connection
        self.shortcut = nn.Sequential()

        # If strided convolution is used or if the number of input channels does not match the output for residual connection
        # then the shortcut connection should modify the channel number
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_forward(x) + self.shortcut(x))

    
# If the model depth is more than 50 layers, the BottleNeck is used
# In this project, the ResNet34 will be used. So BottleNeck block not be used.
class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, middle_channels, stride=1):
        super().__init__()
        
        self.residual_forward = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != middle_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_forward(x) + self.shortcut(x))

# double convolution block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(),
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        return self.doubleconv(x)

    
class UResnet(nn.Module):
    def __init__(self, block, layers, num_classes, input_channels=3):
        super().__init__()
        nb_filter = [64, 128, 256, 512, 1024]

        self.in_channel = nb_filter[0]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(block,nb_filter[1], layers[0], 1)
        self.conv2_0 = self._make_layer(block,nb_filter[2], layers[1], 1)
        self.conv3_0 = self._make_layer(block,nb_filter[3], layers[2], 1)
        self.conv4_0 = self._make_layer(block,nb_filter[4], layers[3], 1)
        self.dropout = nn.Dropout(p=0.2)
        self.conv3_1 = DoubleConv((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3], nb_filter[3] * block.expansion)
        self.conv2_2 = DoubleConv((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2], nb_filter[2] * block.expansion)
        self.conv1_3 = DoubleConv((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1], nb_filter[1] * block.expansion)
        self.conv0_4 = DoubleConv(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, middle_channel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, middle_channel, stride))
            self.in_channel = middle_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x1_0 = self.dropout(x1_0)
        x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0 = self.dropout(x2_0)
        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = self.dropout(x3_0)
        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0 = self.dropout(x4_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

