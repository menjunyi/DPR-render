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

# we define Hour Glass network based on the paper
# Stacked Hourglass Networks for Human Pose Estimation
#       Alejandro Newell, Kaiyu Yang, and Jia Deng
# the code is adapted from
# https://github.com/umich-vl/pose-hg-train/blob/master/src/models/hg.lua


def conv3X3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)
# define the network
class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.conv1 = conv3X3(inplanes, outplanes, 1)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = conv3X3(outplanes, outplanes, 1)
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

class HourglassBlock(nn.Module):
    def __init__(self, inplane, middleNet):
        super(HourglassBlock, self).__init__()
        # upper branch
        self.upper = BasicBlock(inplane, inplane)
        
        # lower branch
        self.downSample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upSample = nn.Upsample(scale_factor=2, mode='nearest')
        self.low1 = BasicBlock(inplane, 2*inplane)
        self.low2 = middleNet
        self.low3 = BasicBlock(2*inplane, inplane)
    def forward(self, x):
        out_upper = self.upper(x)
        out_lower = self.downSample(x)
        out_lower = self.low1(out_lower)
        out_lower = self.low2(out_lower)
        out_lower = self.low3(out_lower)
        out_lower = self.upSample(out_lower)
        out = out_lower + out_upper
        return out

class HourglassNet(nn.Module):
    '''
    	basic idea: low layers are shared, upper layers are different	
    	            lighting should be estimated from the inner most layer
        NOTE: we split the bottle neck layer into albedo, normal and lighting
    '''
    def __init__(self, numLight):
        super(HourglassNet, self).__init__()
        self.HG3_nFilter = 64
        self.HG2_nFilter = 128
        self.HG1_nFilter = 256
        self.HG0_nFilter = 512
        
        self.conv1 = nn.Conv2d(12, self.HG3_nFilter, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(self.HG3_nFilter)
        
        # HG3
        self.HG3_up_albedo = BasicBlock(self.HG3_nFilter, self.HG3_nFilter)
        self.HG3_up_normal = BasicBlock(self.HG3_nFilter, self.HG3_nFilter)
        self.HG3_up_light = BasicBlock(self.HG3_nFilter, self.HG3_nFilter)
        self.HG3_downSample = nn.MaxPool2d(kernel_size=2,stride=2)
        self.HG3_upSample_albedo = nn.Upsample(scale_factor=2, mode='bilinear')
        self.HG3_upSample_normal = nn.Upsample(scale_factor=2, mode='bilinear')
        self.HG3_upSample_light = nn.Upsample(scale_factor=2, mode='bilinear')
        self.HG3_low1 = BasicBlock(self.HG3_nFilter, 2*self.HG3_nFilter)
        self.HG3_low2_normal = BasicBlock(2*self.HG3_nFilter, self.HG3_nFilter)
        self.HG3_low2_albedo = BasicBlock(2*self.HG3_nFilter, self.HG3_nFilter)
        self.HG3_low2_light = BasicBlock(2*self.HG3_nFilter, self.HG3_nFilter)
        
        # HG2
        self.HG2_up_albedo = BasicBlock(self.HG2_nFilter, self.HG2_nFilter)
        self.HG2_up_normal = BasicBlock(self.HG2_nFilter, self.HG2_nFilter)
        self.HG2_up_light = BasicBlock(self.HG2_nFilter, self.HG2_nFilter)
        self.HG2_downSample = nn.MaxPool2d(kernel_size=2,stride=2)
        self.HG2_upSample_albedo = nn.Upsample(scale_factor=2, mode='bilinear')
        self.HG2_upSample_normal = nn.Upsample(scale_factor=2, mode='bilinear')
        self.HG2_upSample_light = nn.Upsample(scale_factor=2, mode='bilinear')
        self.HG2_low1 = BasicBlock(self.HG2_nFilter, 2*self.HG2_nFilter)
        self.HG2_low2_normal = BasicBlock(2*self.HG2_nFilter, self.HG2_nFilter)
        self.HG2_low2_albedo = BasicBlock(2*self.HG2_nFilter, self.HG2_nFilter)
        self.HG2_low2_light = BasicBlock(2*self.HG2_nFilter, self.HG2_nFilter)
        
        # HG1
        self.HG1_up_albedo = BasicBlock(self.HG1_nFilter, self.HG1_nFilter)
        self.HG1_up_normal = BasicBlock(self.HG1_nFilter, self.HG1_nFilter)
        self.HG1_up_light = BasicBlock(self.HG1_nFilter, self.HG1_nFilter)
        self.HG1_downSample = nn.MaxPool2d(kernel_size=2,stride=2)
        self.HG1_upSample_albedo = nn.Upsample(scale_factor=2, mode='bilinear')
        self.HG1_upSample_normal = nn.Upsample(scale_factor=2, mode='bilinear')
        self.HG1_upSample_light = nn.Upsample(scale_factor=2, mode='bilinear')
        self.HG1_low1 = BasicBlock(self.HG1_nFilter, 2*self.HG1_nFilter)          
        self.HG1_low2_normal = BasicBlock(2*self.HG1_nFilter, self.HG1_nFilter)
        self.HG1_low2_albedo = BasicBlock(2*self.HG1_nFilter, self.HG1_nFilter)
        self.HG1_low2_light = BasicBlock(2*self.HG1_nFilter, self.HG1_nFilter)
        
        self.inner = BasicBlock(self.HG0_nFilter, self.HG0_nFilter)

        #------------------------------------------------------------------- 
        # for albedo and normal layer, add three convolutional layers to 
        # predict the final result
        #------------------------------------------------------------------- 
        # albedo layer
        self.albedo_conv_1 = nn.Conv2d(self.HG3_nFilter, self.HG3_nFilter/2, kernel_size=3, stride=1, padding=1)
        self.albedo_bn_1 = nn.BatchNorm2d(self.HG3_nFilter/2)
        self.albedo_conv_2 = nn.Conv2d(self.HG3_nFilter/2, 3, kernel_size=3, stride=1, padding=1)
        self.albedo_bn_2 = nn.BatchNorm2d(3)
        self.albedo_conv_3 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        
        # normal layer
        self.normal_conv_1 = nn.Conv2d(self.HG3_nFilter, self.HG3_nFilter/2, kernel_size=3, stride=1, padding=1)
        self.normal_bn_1 = nn.BatchNorm2d(self.HG3_nFilter/2)
        self.normal_conv_2 = nn.Conv2d(self.HG3_nFilter/2, 3, kernel_size=3, stride=1, padding=1)
        self.normal_bn_2 = nn.BatchNorm2d(3)
        self.normal_conv_3 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        
        # for lighting
        self.light_conv_1 = nn.Conv2d(self.HG3_nFilter, 27, kernel_size=3, stride=1, padding=1)
        self.light_bn_1 = nn.BatchNorm2d(27)
        self.light_conv_2 = nn.Conv2d(27, 27, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        feat = self.conv1(x)
        feat = F.relu(self.bn1(feat))
        # get the inner most features
        feat_3 = self.HG3_low1(self.HG3_downSample(feat))
        feat_2 = self.HG2_low1(self.HG2_downSample(feat_3))
        feat_1 = self.HG1_low1(self.HG1_downSample(feat_2))
        # split bottle neck layer into albedo, normal and lighting
        feat_inner = self.inner(feat_1)
        
        # get albedo
        feat_albedo_1 = self.HG1_upSample_albedo(self.HG1_low2_albedo(feat_inner)) \
        				+ self.HG1_up_albedo(feat_2)
        feat_albedo_2 = self.HG2_upSample_albedo(self.HG2_low2_albedo(feat_albedo_1)) \
        				+ self.HG2_up_albedo(feat_3)
        feat_albedo_3 = self.HG3_upSample_albedo(self.HG3_low2_albedo(feat_albedo_2)) \
        				+ self.HG3_up_albedo(feat)
        albedo = F.relu(self.albedo_bn_1(self.albedo_conv_1(feat_albedo_3)))
        albedo = F.relu(self.albedo_bn_2(self.albedo_conv_2(albedo)))
        albedo = F.sigmoid(self.albedo_conv_3(albedo))
        
        # get normal
        feat_norm_1 = self.HG1_upSample_normal(self.HG1_low2_normal(feat_inner)) \
        				+ self.HG1_up_normal(feat_2)
        feat_norm_2 = self.HG2_upSample_normal(self.HG2_low2_normal(feat_norm_1)) \
        				+ self.HG2_up_normal(feat_3)
        feat_norm_3 = self.HG3_upSample_normal(self.HG3_low2_normal(feat_norm_2)) \
        				+ self.HG3_up_normal(feat)
        normal = F.relu(self.normal_bn_1(self.normal_conv_1(feat_norm_3)))
        normal = F.relu(self.normal_bn_2(self.normal_conv_2(normal)))
        normal = self.normal_conv_3(normal)
        
        # get light
        feat_light_1 = self.HG1_upSample_light(self.HG1_low2_light(feat_inner)) \
        				+ self.HG1_up_light(feat_2)
        feat_light_2 = self.HG2_upSample_light(self.HG2_low2_light(feat_light_1)) \
        				+ self.HG2_up_light(feat_3)
        feat_light_3 = self.HG3_upSample_light(self.HG3_low2_light(feat_light_2)) \
        				+ self.HG3_up_light(feat)
        light = F.relu(self.light_bn_1(self.light_conv_1(feat_light_3)))
        light = self.light_conv_2(light)

        return albedo, normal, light

if __name__ == '__main__':
		net = HourglassNet()
		x = Variable(torch.Tensor(1, 3, 224, 224))
		albedo, norm, lighting, shading = net(x)
		g = make_dot(shading)
		g.render('HourglassNet')
