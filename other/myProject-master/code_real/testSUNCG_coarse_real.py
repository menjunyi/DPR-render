import sys
sys.path.append('../utils')
sys.path.append('../model')
import options
import importlib
from utils_loadModel import *
from collections import OrderedDict

# get input argument
args = options.options()
args.initialize()
args = args.parser.parse_args()
print args.coarseModel
# load model
coarseModel, fineModel, detailModel = loadModels(args)

from skimage import io

import os
import numpy as np
import time

import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch
from tensorboardX import SummaryWriter

# set random seed for all possible randomness
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from setDataPath import *
from network_forward import *
from utils_toGPU import *
from utils_shading import *
from defineHelp_lighting import *
from defineHelp import *
from defineCriterion import *
from defineLoss_IIW import *
from defineLoss_SAW import *
from evaluate_network import *

# define help layer
shadingLayer_coarse = constructShading()
shadingLayer_coarse.cuda()
gradientLayer = gradientLayer_color()
gradientLayer.cuda()
samplingLightLayer = samplingLight()
samplingLightLayer.cuda()

# define criterion
criterion_mask = mask_loss()
criterion_IIW = WhdrHingeLoss()
#criterion_IIW = WhdrTestLoss_Paper()
criterion_SAW_1 = loss_SAW_1()
criterion_SAW_2 = loss_SAW_2()
evaluateNetwork = evaluateNetwork(criterion_mask, criterion_IIW, 
        criterion_SAW_1, criterion_SAW_2, gradientLayer)

## for lighting
# get the normal/direction of positive and negative spheres
# used to constrain lighting to be positive
sphereDirection_pos, sphereDirection_neg, sphereMask = getSHNormal(64)
sphereDirection_pos = Variable(torch.from_numpy(sphereDirection_pos).cuda()).float()
sphereDirection_neg = Variable(torch.from_numpy(sphereDirection_neg).cuda()).float()
sphereMask = Variable(torch.from_numpy(sphereMask).cuda()).float()

# define network forward for coarse model
network = networkForward_basic(coarseModel, shadingLayer_coarse, args.imageSize, 1)
network.model.cuda()

# test loader for IIW evaluation
testLoaderHelper_IIW = constructDataLoader('test', 'compare', args.imageSize)
testLoader_IIW = torch.utils.data.DataLoader(testLoaderHelper_IIW, 
	batch_size=args.batchSize, shuffle=False, num_workers=1)

# test loader for SAW evaluation
# NOTE: the biggest difference is data augmentation
# For comparison, we cannot rotate the image, for SAW, we could rotate 
# the image
testLoaderHelper_SAW = constructDataLoader('test', 'shading', args.imageSize)
testLoader_SAW = torch.utils.data.DataLoader(testLoaderHelper_SAW,
	batch_size=args.batchSize, shuffle=False, num_workers=1)

def help_computeNetwork(data_compare, data_SAW, i, indType='train'):
    # return loss
    # print i
    #begin_time = time.time()
    
    return_loss = OrderedDict()
    image_SUNCG_compare, image_IIW_compare, _, label_SUNCG_compare, \
            label_IIW_compare, _ = data_compare 
    # train on compare 
    images_SUNCG_compare = Variable(image_SUNCG_compare.cuda()).float()
    output_SUNCG_compare = network.forward(images_SUNCG_compare) 
    label_SUNCG_compare = labelToGPU.toGPU(label_SUNCG_compare) 
    loss_SUNCG_compare = evaluateNetwork.IIW_loss(output_SUNCG_compare, label_SUNCG_compare, 'SUNCG', scale_factor=4)
    
    images_IIW_compare = Variable(image_IIW_compare.cuda()).float()
    output_IIW_compare = network.forward(images_IIW_compare) 
    label_IIW_compare = labelToGPU.toGPU(label_IIW_compare) 
    loss_IIW_compare = evaluateNetwork.IIW_loss(output_IIW_compare, label_IIW_compare, 'IIW', scale_factor=4)

    loss = loss_SUNCG_compare['WHDR'] + loss_IIW_compare['WHDR']

    return_loss['SUNCG_WHDR'] = loss_SUNCG_compare['WHDR'].data[0]
    return_loss['IIW_WHDR'] = loss_IIW_compare['WHDR'].data[0]
    
    # backward for IIW compare loss
    if indType=='train':
        loss.backward()
    #print 'time used for one WHDR %s' % (time.time() - begin_time)
    #begin_time = time.time()
    
    image_SUNCG_SAW, image_IIW_SAW, image_NYU_SAW, label_SUNCG_SAW, \
            label_IIW_SAW, label_NYU_SAW = data_SAW
    # suncg SAW 
    #image_SUNCG_SAW, image_IIW_SAW, image_NYU_SAW, label_SUNCG_SAW, \
    #        label_IIW_SAW, label_NYU_SAW = data_compare
    # suncg SAW 
    images_SUNCG_SAW = Variable(image_SUNCG_SAW.cuda()).float()
    output_SUNCG_SAW = network.forward(images_SUNCG_SAW) 
    label_SUNCG_SAW = labelToGPU.toGPU(label_SUNCG_SAW) 
    loss_SUNCG_SAW = evaluateNetwork.SUNCG_data(output_SUNCG_SAW, label_SUNCG_SAW)

    loss_light_pos = nonNegativeLighting_coarseLevel(samplingLightLayer, sphereDirection_pos, output_SUNCG_SAW['lighting'])
    loss_light_neg = nonNegativeLighting_coarseLevel(samplingLightLayer, sphereDirection_neg, output_SUNCG_SAW['lighting'])
    loss_light = loss_light_neg + loss_light_pos

    return_loss['SUNCG_albedo'] = loss_SUNCG_SAW['albedo'].data[0]
    #return_loss['SUNCG_albedo_grad'] = loss_SUNCG_SAW['albedo_grad'].data[0]
    return_loss['SUNCG_shading'] = loss_SUNCG_SAW['shading'].data[0]
    #return_loss['SUNCG_shading_grad'] = loss_SUNCG_SAW['shading_grad'].data[0]
    return_loss['SUNCG_normal'] = loss_SUNCG_SAW['normal'].data[0]
    #return_loss['SUNCG_normal_grad'] = loss_SUNCG_SAW['normal_grad'].data[0]
    return_loss['SUNCG_lighting'] = loss_light.data[0]
    
    # IIW SAW 
    #images_IIW_SAW = Variable(image_IIW_SAW.cuda()).float()
    #output_IIW_SAW = network.forward(images_IIW_SAW) 
    #label_IIW_SAW = labelToGPU.toGPU(label_IIW_SAW) 
    #loss_IIW_SAW = evaluateNetwork.IIW_data(output_IIW_SAW, images_IIW_SAW, label_IIW_SAW)
    #return_loss['IIW_SAW_1'] = loss_IIW_SAW['saw_loss_1'].data[0]
    #return_loss['IIW_SAW_2'] = loss_IIW_SAW['saw_loss_2'].data[0]
    #return_loss['IIW_reconstruct'] = loss_IIW_SAW['reconstruct'].data[0]
    
    # NYU SAW 
    #images_NYU_SAW = Variable(image_NYU_SAW.cuda()).float()
    #output_NYU_SAW = network.forward(images_NYU_SAW) 
    #label_NYU_SAW = labelToGPU.toGPU(label_NYU_SAW) 
    #loss_NYU_SAW = evaluateNetwork.NYU_data(output_NYU_SAW, images_NYU_SAW, label_NYU_SAW)
    #return_loss['NYU_SAW_1'] = loss_NYU_SAW['saw_loss_1'].data[0]
    #return_loss['NYU_SAW_2'] = loss_NYU_SAW['saw_loss_2'].data[0]
    #return_loss['NYU_reconstruct'] = loss_NYU_SAW['reconstruct'].data[0]
    #return_loss['NYU_normal'] = loss_NYU_SAW['normal'].data[0]
    #return_loss['NYU_normal_grad'] = loss_NYU_SAW['normal_grad'].data[0]
    
    #loss = loss_SUNCG_SAW['albedo'] + loss_SUNCG_SAW['normal'] + loss_SUNCG_SAW['shading'] \
    #        + loss_SUNCG_SAW['albedo_grad'] + loss_SUNCG_SAW['normal_grad'] + loss_SUNCG_SAW['shading_grad']\
    #        + loss_IIW_SAW['saw_loss_1'] + loss_IIW_SAW['saw_loss_2'] + loss_IIW_SAW['reconstruct'] \
    #        + loss_NYU_SAW['saw_loss_1'] + loss_NYU_SAW['saw_loss_2'] + loss_NYU_SAW['reconstruct'] \
    #        + loss_NYU_SAW['normal'] + loss_NYU_SAW['normal_grad']
    loss = loss_SUNCG_SAW['albedo'] + loss_SUNCG_SAW['normal'] + loss_SUNCG_SAW['shading'] \
            + loss_SUNCG_SAW['albedo_grad'] + loss_SUNCG_SAW['normal_grad'] + loss_SUNCG_SAW['shading_grad'] + loss_light
            #+ loss_IIW_SAW['saw_loss_1'] + loss_IIW_SAW['saw_loss_2'] + loss_IIW_SAW['reconstruct'] \
            #+ loss_NYU_SAW['saw_loss_1'] + loss_NYU_SAW['saw_loss_2'] + loss_NYU_SAW['reconstruct'] \
            #+ loss_NYU_SAW['normal'] + loss_NYU_SAW['normal_grad']


    if indType=='train':
        loss.backward()
        optimizer.step()
    #print 'time used for others %s' % (time.time() - begin_time)

    return return_loss

def record_loss(writer, loss_list, lossName, numIte, indType='train'):
    '''
        write loss into tensorboard
    '''
    for (i, item) in enumerate(lossName):
        writer.add_scalar(os.path.join(indType, item), loss_list[i], numIte)


labelToGPU = labelToGPU()

def main():
    #lossName = ['SUNCG_WDHR', 'IIW_WDHR', 'SUNCG_albedo', 'SUNCG_albedo_grad',
    #        'SUNCG_shading', 'SUNCG_shading_normal', 'SUNCG_normal', 'SUNCG_normal_grad',
    #        'IIW_SAW_1', 'IIW_SAW_2', 'IIW_reconstruct', 'NYU_SAW_1', 'NYU_SAW_2', 'NYU_reconstruct',
    #        'NYU_normal', 'NYU_normal_grad']
    #lossName = ['SUNCG_WDHR_loss', 'IIW_WDHR_loss', 'SUNCG_albedo_loss', 'SUNCG_shading_loss', 'SUNCG_normal_loss']
    lossName = ['SUNCG_WHDR', 'IIW_WHDR', 'SUNCG_albedo_loss', 'SUNCG_shading_loss', 'SUNCG_normal_loss', 'SUNCG_lighting']
    numIte_testing = 0

    test_SAW_iter = iter(testLoader_SAW)

    network.model.train(False)
    test_loss_list = []

    for (i, data_compare) in enumerate(testLoader_IIW, 0): 
        data_SAW = next(test_SAW_iter, None)
        return_loss = help_computeNetwork(data_compare, data_SAW, i, indType='test')
        numLoss = len(return_loss.values())
        test_loss_list.append(return_loss.values())
    test_loss_list = np.array(test_loss_list)
    test_loss_list = np.mean(test_loss_list, axis=0)
    #record_loss(writer, test_loss_list, lossName, numIte_training, indType='test')
    print '%0.6f '* numLoss % tuple(test_loss_list)

if __name__ =='__main__':
    main()
