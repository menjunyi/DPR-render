import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
sys.path.append('/scratch1/intrinsicImage/myProject/code_synthetic/functional-zoo/')
from visualize import make_dot
import time

from utils_shading import *

# according to David Eigen's paper and Xiang's suggestions
# we try a predict normal with normal convolution layers

def conv3X3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=padding, dilation=dilation, bias=False)
# define the network
class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, padding=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.conv1 = conv3X3(inplanes, outplanes, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = conv3X3(outplanes, outplanes, stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(outplanes)
        
        self.shortcuts = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.inplanes != self.outplanes:
        		out += self.shortcuts(x)
        else:
        		out += x
        
        out = F.relu(out)
        return out

class basicNetwork(nn.Module):
    def __init__(self, inplane, outplane):
        super(basicNetwork, self).__init__()
        self.nFilters=64

        self.dilation_1=2
        self.dilation_2=2

        self.conv1 = nn.Conv2d(inplane, self.nFilters, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(self.nFilters)

        # middel level
        self.block1= BasicBlock(self.nFilters, self.nFilters, stride=1, padding=self.dilation_1, dilation=self.dilation_1)
        self.block2= BasicBlock(self.nFilters, self.nFilters, stride=1, padding=self.dilation_2, dilation=self.dilation_2)

        # fine level, 3 1x1 convolutional layers
        self.fine_conv1 = nn.Conv2d(self.nFilters, self.nFilters, kernel_size=1, stride=1, padding=0, bias=False)
        self.fine_bn1 = nn.BatchNorm2d(self.nFilters)
        self.fine_conv2 = nn.Conv2d(self.nFilters, self.nFilters, kernel_size=1, stride=1, padding=0, bias=False)
        self.fine_bn2 = nn.BatchNorm2d(self.nFilters)
        self.fine_conv3 = nn.Conv2d(self.nFilters, self.nFilters, kernel_size=1, stride=1, padding=0, bias=False)
        self.fine_bn3 = nn.BatchNorm2d(self.nFilters)

        # combine everyting
        self.combine_conv1 = nn.Conv2d(self.nFilters*2 + 3, self.nFilters, kernel_size=3, stride=1, padding=1, bias=False)
        self.combine_bn1 = nn.BatchNorm2d(self.nFilters)
        self.combine_conv2 = nn.Conv2d(self.nFilters, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.combine_bn2 = nn.BatchNorm2d(3)
        self.combine_conv3 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x, coarse_normal):
        feat = F.relu(self.bn1(self.conv1(x)))

        feat_1 = self.block1(feat)
        feat_2 = self.block2(feat_1)

        fine_1 = F.relu(self.fine_bn1(self.fine_conv1(feat)))
        fine_2 = F.relu(self.fine_bn2(self.fine_conv2(fine_1)))
        fine_3 = F.relu(self.fine_bn3(self.fine_conv3(fine_2)))

        out = torch.cat((coarse_normal, feat_2, fine_3), dim=1)
        out = F.relu(self.combine_bn1(self.combine_conv1(out)))
        out = F.relu(self.combine_bn2(self.combine_conv2(out)))
        out = F.normalize(self.combine_conv3(out), p=2, dim=1)
        
        return out


class networkNormal(nn.Module):
    def __init__(self, inplane_normal):
        super(networkNormal, self).__init__()
        self.normalNetwork = basicNetwork(inplane_normal, 3)
    
    def forward(self, in_normal, coarse_normal):
        normal = self.normalNetwork(in_normal, coarse_normal)
        return normal

if __name__ == '__main__':
		net = HourglassNet()
		x = Variable(torch.Tensor(1, 3, 224, 224))
		albedo, norm, lighting, shading = net(x)
		g = make_dot(shading)
		g.render('HourglassNet')
