from models.fcn import FCN32s
from models.fcn_vgg import VGGNet, FCN8s
from models.unet import UNet
from models.unet_resnet import UResnet, BasicBlock, BottleNeck
from models.unet_resnet_pretrained import UResnet_pretrained
from models.resunet_a import ResUnet_a_d6
from models.resunet_a_d6_multitask import ResUnet_a_d6_Multitask

# Calculate the total number of parameters of a model
# and the total number of trainable parameters
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

net = UNet(n_channels = 3 , n_classes = 8)

param = get_parameter_number(net)
print(str(param))
#print(net.state_dict())