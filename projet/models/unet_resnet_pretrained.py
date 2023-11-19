import torch
import torch.nn as nn
import torchvision.models as models

# Build my own decoder
class DecoderBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
    
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.norm1 = nn.BatchNorm2d(mid_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

        self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels = out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        

    def forward(self,x):
        x=self.conv(x)
        x=self.norm1(x)
        x=self.relu1(x)
        x=self.upsample(x)
        x=self.norm2(x)
        x=self.relu2(x)
        return x

class UResnet_pretrained(nn.Module):

    def __init__(self, backbone, n_class):
        super().__init__()
        self.backbone = backbone
        self.n_class = n_class

        if backbone=='resnet34': # choose the pretrained Resnet model
            resnet = models.resnet34(pretrained=True)
            filters=[64,64,128,256,512]
        elif backbone=='resnet50':
            resnet = models.resnet50(pretrained=True)
            filters=[64,256,512,1024,2048]
            
#Changing the number of channels and other operations on the input image in order to make it correspond to the input of the encoder of the pretrianed Resnet model.
        self.firstconv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False) #
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.dropout = nn.Dropout(p=0.2)
        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3]*4, out_channels=filters[3])
        self.decoder1 = DecoderBlock(in_channels=filters[3]+filters[2], mid_channels=filters[2]*4, out_channels=filters[2])
        self.decoder2 = DecoderBlock(in_channels=filters[2]+filters[1], mid_channels=filters[1]*4, out_channels=filters[1])
        self.decoder3 = DecoderBlock(in_channels=filters[1]+filters[0], mid_channels=filters[0]*4, out_channels=filters[0])
        self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0],out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), 
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=n_class, kernel_size=1)
                )

    def forward(self,x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)

        e1 = self.encoder1(x_)
        e1 = self.dropout(e1)
        e2 = self.encoder2(e1)
        e2 = self.dropout(e2)
        e3 = self.encoder3(e2)
        e3 = self.dropout(e3)

        center = self.center(e3)

        d2 = self.decoder1(torch.cat([center,e2],dim=1))
        d3 = self.decoder2(torch.cat([d2,e1], dim=1))
        d4 = self.decoder3(torch.cat([d3,x], dim=1))

        return self.final(d4)