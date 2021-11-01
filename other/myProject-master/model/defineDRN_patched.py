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

# we define dilated residual network based on the paper
#              Dilated Residual Networks
# Fisher Yu, Vladlen Koltun and Thomas Funkhouser
# the code is adapted from
# https://github.com/fyu/drn/blob/master/drn.py

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
        self.nFilters_0=16
        self.nFilters_1=32
        self.nFilters_2=32
        self.nFilters_3=64
        self.nFilters_4=32
        self.nFilters_5=32

        self.dilation_1=1
        self.dilation_2=2
        self.dilation_3=4
        self.dilation_4=1
        self.dilation_5=1

        self.conv1 = nn.Conv2d(inplane, self.nFilters_0, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(self.nFilters_0)

        # shared
        self.block_1= BasicBlock(self.nFilters_0, self.nFilters_1, stride=1, padding=1, dilation=self.dilation_1)
        self.block_2= BasicBlock(self.nFilters_1, self.nFilters_2, stride=1, padding=2, dilation=self.dilation_2)
        self.block_3= BasicBlock(self.nFilters_2, self.nFilters_3, stride=1, padding=4, dilation=self.dilation_3)
        self.block_4= BasicBlock(self.nFilters_3, self.nFilters_4, stride=1, padding=1, dilation=self.dilation_4)
        self.block_5= BasicBlock(self.nFilters_4, self.nFilters_5, stride=1, padding=1, dilation=self.dilation_5)
        self.pred_1 = nn.Conv2d(self.nFilters_5, outplane, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(outplane)
        self.pred_2 = nn.Conv2d(outplane, outplane, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        feat = F.relu(self.bn1(self.conv1(x)))
        feat_1 = self.block_1(feat)
        feat_2 = self.block_2(feat_1)
        feat_3 = self.block_3(feat_2)
        feat_4 = self.block_4(feat_3) + feat_2
        feat_5 = self.block_5(feat_4) + feat_1
        
        output = F.relu(self.bn2(self.pred_1(feat_5)))
        output = self.pred_2(output)
        return output

class basicLightingNetwork(nn.Module):
    '''
        dealing with 112x112 images and output 7x7 lighting
        each lighting is the lighting in a region of 16x16
    '''
    def __init__(self, inplane, outplane):
        super(basicLightingNetwork, self).__init__()
        self.nFilters_0=16
        self.nFilters_1=32
        self.nFilters_2=32
        self.nFilters_3=64
        self.nFilters_4=32
        self.nFilters_5=32

        self.dilation_1=1
        self.dilation_2=2
        self.dilation_3=4
        self.dilation_4=1
        self.dilation_5=1

        self.conv1 = nn.Conv2d(inplane, self.nFilters_0, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(self.nFilters_0)

        # shared
        self.block_1= BasicBlock(self.nFilters_0, self.nFilters_1, stride=1, padding=1, dilation=self.dilation_1)
        self.downSample_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block_2= BasicBlock(self.nFilters_1, self.nFilters_2, stride=1, padding=2, dilation=self.dilation_2)
        self.downSample_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block_3= BasicBlock(self.nFilters_2, self.nFilters_3, stride=1, padding=4, dilation=self.dilation_3)
        self.downSample_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block_4= BasicBlock(self.nFilters_3, self.nFilters_4, stride=1, padding=1, dilation=self.dilation_4)
        self.block_5= BasicBlock(self.nFilters_4, self.nFilters_5, stride=1, padding=1, dilation=self.dilation_5)
        self.pred_1 = nn.Conv2d(self.nFilters_5, outplane, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(outplane)
        self.pred_2 = nn.Conv2d(outplane, outplane, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        feat = F.relu(self.bn1(self.conv1(x)))
        feat_1 = self.downSample_1(self.block_1(feat))
        feat_2 = self.downSample_2(self.block_2(feat_1))
        feat_3 = self.downSample_3(self.block_3(feat_2))
        feat_4 = self.block_4(feat_3)
        feat_5 = self.block_5(feat_4)
        
        output = F.relu(self.bn2(self.pred_1(feat_5)))
        output = self.pred_2(output)
        return output


class DRN(nn.Module):
    def __init__(self, inplane_albedo, inplane_normal, inplane_lighting):
        super(DRN, self).__init__()
        self.albedoNetwork = basicNetwork(inplane_albedo, 3)
        self.normalNetwork = basicNetwork(inplane_normal, 3)
        self.lightingNetwork = basicLightingNetwork(inplane_lighting, 27)
    
    def forward(self, in_albedo, in_normal, in_lighting):
        albedo = self.albedoNetwork(in_albedo)
        normal = self.normalNetwork(in_normal)

        # predict lighting for every 16x16 patch
        in_lighting_UpLeft    = in_lighting[:,:, 0:-8, 0:-8]
        in_lighting_UpRight   = in_lighting[:,:, 0:-8, 8:]
        in_lighting_DownLeft  = in_lighting[:,:, 8:, 0:-8]
        in_lighting_DownRight = in_lighting[:,:, 8:, 8:] 

        lighting_UpLeft = self.lightingNetwork(in_lighting_UpLeft)
        lighting_UpRight = self.lightingNetwork(in_lighting_UpRight)
        lighting_DownLeft = self.lightingNetwork(in_lighting_DownLeft)
        lighting_DownRight = self.lightingNetwork(in_lighting_DownRight)
        
        return albedo, normal, lighting_UpLeft, lighting_UpRight, lighting_DownLeft, lighting_DownRight

if __name__ == '__main__':
		net = HourglassNet()
		x = Variable(torch.Tensor(1, 3, 224, 224))
		albedo, norm, lighting, shading = net(x)
		g = make_dot(shading)
		g.render('HourglassNet')
