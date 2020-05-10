from sync_bn_lib.sync_batchnorm import SynchronizedBatchNorm2d
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from lib.constants import no_of_classes
from lib.gaussian_layer import GaussianKernel
from lib.utils import recursive_iterate_modules


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class ResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        self.net = models.resnet18(pretrained)
        # Freeze the resnet model
        for name, param in self.net.named_parameters():
            if "bn" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        for module in recursive_iterate_modules(self.net):
            if isinstance(module, torch.nn.BatchNorm2d):
                module.momentum = 0.02
 
    def forward(self, x):
        out = {}
        for name, module in self.net._modules.items():
            if name == "fc":
                x = x.view(x.size(0), -1)
            x = module(x)
            out[name] = x
        return out


class GaussianModel(nn.Module):

    def __init__(self, pretrained_net):
        super(GaussianModel, self).__init__()
        self.pretrained_net = pretrained_net

        self.deconv4 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.deconv4_bn = SynchronizedBatchNorm2d(256, momentum=0.02)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.deconv3_bn = SynchronizedBatchNorm2d(128, momentum=0.02)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv2_bn = SynchronizedBatchNorm2d(64, momentum=0.02)
        self.deconv1 = nn.ConvTranspose2d(64, 48, 3, stride=2, padding=1, output_padding=1)
        self.deconv1_bn = SynchronizedBatchNorm2d(48, momentum=0.02)
        self.mid_gaussian = GaussianKernel(in_count=48)
        self.deconv = nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1)
        self.deconv_bn = SynchronizedBatchNorm2d(48, momentum=0.02)
        self.final_gaussian = GaussianKernel(in_count=48)
        self.conv11 = nn.Conv2d(48, no_of_classes, 1)

        self.prelu4 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu1 = nn.PReLU()
        self.prelu = nn.PReLU()


    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output["layer4"]
        x4 = output["layer3"]
        x3 = output["layer2"]
        x2 = output["layer1"]
        
        x = x5
        x = self.prelu4(self.deconv4(x))
        x = self.deconv4_bn(x + x4)
        x = self.prelu3(self.deconv3(x))
        x = self.deconv3_bn(x + x3)
        x = self.prelu2(self.deconv2(x))
        x = self.deconv2_bn(x + x2)
        x = self.deconv1_bn(self.prelu1(self.deconv1(x)))

        x = self.mid_gaussian(x)
        x = self.deconv_bn(self.prelu(self.deconv(x)))

        x = self.final_gaussian(x)
        x = self.conv11(x)
        return x
