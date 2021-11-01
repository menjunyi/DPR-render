'''
    this file is used to precompute the albedo, normal and shading for coarse scale
'''
import sys
sys.path.append('utils')
sys.path.append('model')

import os
import numpy as np
import time
import imageio

from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F

# load module defined by myself
from defineHourglass_64 import *
from utils_shading import *
from defineCriterion import *
from loadData_basic import *
from defineHelp import *
from defineHelp_lighting import *

# set random seed for all possible randomness
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


baseFolder = '/scratch1/intrinsicImage/synthetic_SUNCG/'

#criterion = MSELoss()
criterion = mask_loss()

gradientLayer = gradientLayer_color()
gradientLayer.cuda()
shadingLayer = constructShading()
shadingLayer.cuda()
samplingLightLayer = samplingLight()
samplingLightLayer.cuda()

transformer_test = transforms.Compose([testTransfer(output_size=64), ToTensor()])
testLoaderHelper = SUNCG(dataFolder=os.path.join(baseFolder, 'images_color'),
    albedoFolder=os.path.join(baseFolder, 'albedo'),
    shadingFolder=os.path.join(baseFolder, 'shading'),
    normalFolder='/scratch1/data/SUNCG_groundTruth/SUNCG_normal',
    maskFolder=os.path.join(baseFolder, 'images_mask_correct'),
    fileListName=os.path.join(baseFolder, 'validFile.list'),
    transform = transformer_test)

testLoader = torch.utils.data.DataLoader(testLoaderHelper, 
    batch_size=20, shuffle=False, num_workers=5)

def getImage(image, output_albedo, output_shading, output_lighting, output_normal, count, resultPath, fileName):
    '''
        compute the loss for the mini-batch
    '''
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    
    for i in range(output_albedo.size()[0]):

        tmpName = fileName[i]
        folder = tmpName.split('/')[0]
        name = tmpName.split('/')[1].split('_')[0]
        fileFolder = os.path.join(resultPath, folder)
        if not os.path.exists(fileFolder):
            os.makedirs(fileFolder)

        imgAlbedo = np.squeeze(output_albedo[i].cpu().data.numpy())
        imgShading = np.squeeze(output_shading[i].cpu().data.numpy())
        imgNormal = np.squeeze(output_normal[i].cpu().data.numpy())
        lighting = np.squeeze(output_lighting[i].cpu().data.numpy())
        
        imgAlbedo = imgAlbedo.transpose((1,2,0))
        imgShading = imgShading.transpose((1,2,0))
        imgNormal = imgNormal.transpose((1,2,0))

        imageio.imsave(os.path.join(fileFolder, name + '_shading.tiff'), imgShading)
        imageio.imsave(os.path.join(fileFolder, name + '_albedo.tiff'), imgAlbedo)
        imageio.imsave(os.path.join(fileFolder, name + '_normal.tiff'), imgNormal)
        fid = open(os.path.join(fileFolder, name + '_lighting.txt'), 'w')
        print>>fid, '%0.8f '*27 % tuple(lighting[0:27])


def main(savePath, modelPath):
    begin_time = time.time()
    my_network = torch.load(os.path.join(modelPath, 'trained_model.t7'))
    my_network.cuda()
    my_network.train(False)
    test_loss = 0
    count = 0
    
    for i, data in enumerate(testLoader, 0):
        #if i != 0:
        #    # for debug
        #    break
        inputs, albedo, shading, normal, mask, fileName = data
        inputs, albedo, shading, normal, mask = Variable(inputs.cuda(), volatile=True).float(), \
            Variable(albedo.cuda(), volatile=True).float(), \
            Variable(shading.cuda(), volatile=True).float(), \
            Variable(normal.cuda(), volatile=True).float(), \
            Variable(mask.cuda(), volatile=True).float()
        #print fileName
        output_albedo, output_normal, output_lighting = my_network(inputs)
        output_shading = shadingLayer(output_normal, output_lighting)

        numSamples = inputs.size()[0]

        getImage(inputs, output_albedo, output_shading, output_lighting, output_normal, count, savePath, fileName)

if __name__ == '__main__':
    savePath = sys.argv[1]
    modelPath = sys.argv[2]
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    main(savePath, modelPath)
