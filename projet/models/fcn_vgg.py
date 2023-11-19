import torch.nn as nn
from torchvision.models.vgg import VGG
from torchvision import models

#  -------------------------Net------------------------------------------

# configuration of a VGGNet
ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# layer builder according to configuration
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# use the VGGNet pretrained model as encoder
class VGGNet(VGG):
    def __init__(self, pretrained = True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained: # laod pretrained weight
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc: #delete claissifier, replace by FCN
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        for idx, (begin, end) in enumerate(self.ranges):
            for layer in range(begin, end):
                x = self.features[layer](x)
            output["x%d" %(idx +1)] = x
        return output

class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        
       # self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
       # self.relu5_1 = nn.ReLU(inplace=True)
       # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
       # self.relu5_2 = nn.ReLU(inplace=True)
       # self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
       # self.relu5_3 = nn.ReLU(inplace=True)
       # self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1)
        self.drop    = nn.Dropout(0.5, inplace=False)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']    
        x4 = output['x4']    
        x3 = output['x3']    

        score = self.relu(self.conv6(x5))   
        score = self.relu(self.conv7(score)) 
        score = self.relu(self.deconv1(score))   
        score = self.bn1(score + x4)
        score = self.relu(self.deconv2(score)) 
        score = self.bn2(score + x3)
        score = self.bn3(self.relu(self.deconv3(score)))  
        score = self.bn4(self.relu(self.deconv4(score)))  
        score = self.bn5(self.relu(self.deconv5(score)))  
        score = self.classifier(score)                    
        
        
        return score