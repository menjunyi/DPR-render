from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from defineNetworks_overlapping import *

'''
    define help functions for defineDRN_patched
'''

def constructWeight(imageSize, patchSize, stride):
    '''
        used to construct weight for patches
    '''
    H_patchSize = patchSize
    W_patchSize = patchSize
    H_stride = stride
    W_stride = stride

    patchWeight = getPatchWeight(H_patchSize, W_patchSize, 
        H_stride, W_stride)
    patchWeightUL = getPatchWeight(H_patchSize, W_patchSize, 
        H_stride, W_stride, boarderUP=True, boarderLEFT=True)
    patchWeightU = getPatchWeight(H_patchSize, W_patchSize, 
        H_stride, W_stride, boarderUP=True)
    patchWeightUR= getPatchWeight(H_patchSize, W_patchSize, 
        H_stride, W_stride, boarderUP=True, boarderRIGHT=True)
    patchWeightR = getPatchWeight(H_patchSize, W_patchSize, 
        H_stride, W_stride, boarderRIGHT=True)
    patchWeightDR = getPatchWeight(H_patchSize, W_patchSize, 
        H_stride, W_stride, boarderRIGHT=True, boarderDOWN=True)
    patchWeightDL = getPatchWeight(H_patchSize, W_patchSize, 
        H_stride, W_stride, boarderLEFT=True, boarderDOWN=True)
    patchWeightD = getPatchWeight(H_patchSize, W_patchSize, 
        H_stride, W_stride, boarderDOWN=True)
    patchWeightL = getPatchWeight(H_patchSize, W_patchSize, 
        H_stride, W_stride, boarderLEFT=True)

    # construct four patches
    numW = (imageSize - (W_patchSize - W_stride))/W_patchSize
    numH = (imageSize - (H_patchSize - H_stride))/H_patchSize
    print numW
    print numH

    # different weight for four images
    imageWeight_UL = np.zeros((imageSize, imageSize))
    imageWeight_UR = np.zeros((imageSize, imageSize))
    imageWeight_DL = np.zeros((imageSize, imageSize))
    imageWeight_DR = np.zeros((imageSize, imageSize))

    # for upper left patch
    for i in range(numH):
        for j in range(numW):
                if i == 0 and j == 0:
                    tmp_patchWeight = patchWeightUL
                elif i == 0:
                    tmp_patchWeight = patchWeightU
                elif j == 0:
                    tmp_patchWeight = patchWeightL
                else:
                    tmp_patchWeight = patchWeight
                imageWeight_UL[i*H_patchSize:(i+1)*H_patchSize, j*W_patchSize:(j+1)*W_patchSize] = \
                        np.copy(tmp_patchWeight)
    # for upper right patch
    for i in range(numH):
        for j in range(numW):
                if i == 0 and j == numW-1:
                    tmp_patchWeight = patchWeightUR
                elif i == 0:
                    tmp_patchWeight = patchWeightU
                elif j == numW - 1:
                    tmp_patchWeight = patchWeightR
                else:
                    tmp_patchWeight = patchWeight
                imageWeight_UR[i*H_patchSize: (i+1)*H_patchSize, 
                        W_stride + j*W_patchSize:W_stride+(j+1)*W_patchSize] = np.copy(tmp_patchWeight)

    # for down left patch
    for i in range(numH):
        for j in range(numW):
                if i == numH -1 and j == 0:
                    tmp_patchWeight = patchWeightDL
                elif j == 0:
                    tmp_patchWeight = patchWeightL
                elif i == numW - 1:
                    tmp_patchWeight = patchWeightD
                else:
                    tmp_patchWeight = patchWeight
                imageWeight_DL[H_stride+i*H_patchSize: H_stride+(i+1)*H_patchSize, 
                        j*W_patchSize:(j+1)*W_patchSize] = np.copy(tmp_patchWeight)
    # for down right patch
    for i in range(numH):
        for j in range(numW):
                if i == numH - 1 and j == numW - 1:
                    tmp_patchWeight = patchWeightDR
                elif i == numW - 1:
                    tmp_patchWeight = patchWeightD
                elif j == numW - 1:
                    tmp_patchWeight = patchWeightR
                else:
                    tmp_patchWeight = patchWeight
                imageWeight_DR[H_stride + i*H_patchSize: H_stride + (i+1)*H_patchSize, 
                        W_stride+j*W_patchSize:W_stride + (j+1)*W_patchSize] = np.copy(tmp_patchWeight)
    return imageWeight_UL, imageWeight_UR, imageWeight_DL, imageWeight_DR



class constructShading(nn.Module):
    def __init__(self, imageSize, patchSize, stride):
        super(constructShading, self).__init__()
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.stride = stride

        # create the weight for combining patches with overlap region
        # four corners are only covered by one patch
        # four strips on the boundary are covered by 2 patches
        # inside regions are covered by four patches
        self.weight_UL, self.weight_UR, self.weight_DL, self.weight_DR = \
                constructWeight(imageSize, patchSize, stride)
        self.weight_UL = np.tile(self.weight_UL[None,None,...], (1,3,1,1))
        self.weight_UR = np.tile(self.weight_UR[None,None,...], (1,3,1,1))
        self.weight_DL = np.tile(self.weight_DL[None,None,...], (1,3,1,1))
        self.weight_DR = np.tile(self.weight_DR[None,None,...], (1,3,1,1))

        self.weight_UL = Variable(torch.from_numpy(self.weight_UL).cuda()).float()
        self.weight_UR = Variable(torch.from_numpy(self.weight_UR).cuda()).float()
        self.weight_DL = Variable(torch.from_numpy(self.weight_DL).cuda()).float()
        self.weight_DR = Variable(torch.from_numpy(self.weight_DR).cuda()).float()

        # for lighting
        self.att = Variable(torch.Tensor(np.pi*np.array([1, 2.0/3, 0.25])).cuda()).float()
        self.help_1 = Variable(torch.Tensor([np.sqrt(1.0/(4*np.pi))]).cuda()).float()
        self.help_2 = Variable(torch.Tensor([np.sqrt(3.0/(4*np.pi))]).cuda()).float()
        self.help_3 = Variable(torch.Tensor([np.sqrt(5.0/(4*np.pi))]).cuda()).float()
        self.help_4 = Variable(torch.Tensor([np.sqrt(5.0/(12*np.pi))]).cuda()).float()
        self.help_5 = Variable(torch.Tensor([np.sqrt(5.0/(48*np.pi))]).cuda()).float()
        self.numImages = -1


    def getSH(self, normal):
        '''
        	from normal to SH basis
        	supposing normal has the format:
        	N  x 3 X x Y 
        '''
        
        x = torch.unsqueeze(normal[:,0,:,:], -1)
        y = torch.unsqueeze(normal[:,1,:,:], -1)
        z = torch.unsqueeze(normal[:,2,:,:], -1)
        
        # for normals with all elements being 0
        # we set it to be zero
        tmpOnes = x**2 + y**2 + z**2
        sh_0 = tmpOnes.float()*self.att[0]*self.help_1
        sh_1 = self.att[1]*self.help_2*z
        sh_2 = self.att[1]*self.help_2*x
        sh_3 = self.att[1]*self.help_2*y
        sh_4 = self.att[2]*0.5*self.help_3*(2*z*z-x*x-y*y)
        sh_5 = self.att[2]*(3.0*self.help_4)*(x*z)
        sh_6 = self.att[2]*(3.0*self.help_4)*(y*z)
        sh_7 = self.att[2]*(3.0*self.help_5)*(x*x-y*y)
        sh_8 = self.att[2]*(3.0*self.help_4)*(x*y)
        sh = torch.cat((sh_0, sh_1, sh_2, sh_3, sh_4, sh_5, sh_6, sh_7, sh_8), -1)
        return sh
    
    def getPatchShading(self, lighting, sh_normal):
        '''
            contrusct shading 
        '''
        lighting = lighting.permute(0,2,3,1)
        numImages = sh_normal.shape[0]
        num_x = sh_normal.shape[1]
        num_y = sh_normal.shape[2]
        tmpShading = Variable(torch.zeros(numImages, num_x, num_y, 3).cuda()).float()
        light_0 = lighting[:, :, :, 0:9]
        light_1 = lighting[:, :, :, 9:18]
        light_2 = lighting[:, :, :, 18:27]
        
        tmpShading[:, :, :, 0] = torch.sum(sh_normal*light_0, dim=-1)
        tmpShading[:, :, :, 1] = torch.sum(sh_normal*light_1, dim=-1)
        tmpShading[:, :, :, 2] = torch.sum(sh_normal*light_2, dim=-1)
        return tmpShading.permute(0, 3, 1, 2)

    def expandLighting(self, lighting):
        '''
            exapnd lighting to match the size of the image
            for exampl, lighting here is a nx27x7x7 matrix
            I want to expand it to be nx27x112x112, so each 
            16x16 patch in the expanded lighting matrix has 
            for all its pixels and the value if copied from
            correponding 7x7 patch
        '''
        numImages = lighting.shape
        num_Height = lighting.shape[2]
        num_Width = lighting.shape[3]

        expand_lighting  = Variable(torch.zeros(self.numImages, 27, 
            self.imageSize - self.stride, self.imageSize - self.stride).cuda()).float()

        for i in range(num_Height):
            for j in range(num_Width):
                tmp_light = lighting[:,:,i,j]
                # expand lighting to nxcximageSizeximageSize
                tmp_light = tmp_light.repeat(self.patchSize*self.patchSize, 1)
                tmp_light = tmp_light.view(self.patchSize, self.patchSize, self.numImages, -1)
                expand_lighting[:,:,i*self.patchSize:(i+1)*self.patchSize,
                        j*self.patchSize:(j+1)*self.patchSize] = tmp_light.permute(2,3,0,1)
        return expand_lighting


    def forward(self, light_UL, light_UR, light_DL, light_DR, normal):
        self.numImages = normal.shape[0]

        sh = self.getSH(normal)
        num_x = normal.shape[2]
        num_y = normal.shape[3]
        shading = Variable(torch.zeros(self.numImages, 3, num_x, num_y).cuda()).float()
        tmp_UL = Variable(torch.zeros(self.numImages, 3, num_x, num_y).cuda()).float()
        tmp_UR = Variable(torch.zeros(self.numImages, 3, num_x, num_y).cuda()).float()
        tmp_DL = Variable(torch.zeros(self.numImages, 3, num_x, num_y).cuda()).float()
        tmp_DR = Variable(torch.zeros(self.numImages, 3, num_x, num_y).cuda()).float()

        # Up left
        expand_UL = self.expandLighting(light_UL)
        tmp_UL[:,:,0:-8, 0:-8] = self.getPatchShading(expand_UL, sh[:,0:-8, 0:-8,:])
        # Up right
        expand_UR = self.expandLighting(light_UR)
        tmp_UR[:,:,0:-8, 8:] = self.getPatchShading(expand_UR, sh[:,0:-8, 8:,:])
        # down left
        expand_DL = self.expandLighting(light_DL)
        tmp_DL[:,:,8:, 0:-8] = self.getPatchShading(expand_DL, sh[:,8:, 0:-8,:])
        # down righ
        expand_DR = self.expandLighting(light_DR)
        tmp_DR[:,:,8:, 8:] = self.getPatchShading(expand_DR, sh[:,8:, 8:,:])

        shading = tmp_UL*self.weight_UL.repeat(self.numImages, 1,1,1) \
                + tmp_UR*self.weight_UR.repeat(self.numImages, 1,1,1) \
                + tmp_DL*self.weight_DL.repeat(self.numImages, 1,1,1) \
                + tmp_DR*self.weight_DR.repeat(self.numImages, 1,1,1)
        return shading
