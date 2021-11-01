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

from setDataPath_fileName import *
from network_forward import *
from utils_toGPU import *
from utils_shading import *
from defineHelp_lighting import *
from defineHelp import *
from defineCriterion import *
from defineLoss_WHDR_symmetric import *
from defineLoss_SAW import *
from evaluate_network import *

# define help layer
shadingLayer_coarse = constructShading()
shadingLayer_coarse.cuda()
gradientLayer = gradientLayer_color()
gradientLayer.cuda()
shadingLayer = constructShading_lightImg()
shadingLayer.cuda()
samplingLightLayer = samplingLight()
samplingLightLayer.cuda()

# define criterion
criterion_mask = mask_loss()
criterion_IIW = WhdrHingeLoss(margin=0.08, delta=0.12)
criterion_IIW_paper = WhdrTestLoss_Paper(delta=0.1)
criterion_SAW_1 = loss_SAW_1()
criterion_SAW_2 = loss_SAW_2()
evaluateNetwork = evaluateNetwork(criterion_mask, criterion_IIW, 
        criterion_IIW_paper, criterion_SAW_1, criterion_SAW_2, gradientLayer)

## for lighting
# get the normal/direction of positive and negative spheres
# used to constrain lighting to be positive
sphereDirection_pos, sphereDirection_neg, sphereMask = getSHNormal(64)
sphereDirection_pos = Variable(torch.from_numpy(sphereDirection_pos).cuda()).float()
sphereDirection_neg = Variable(torch.from_numpy(sphereDirection_neg).cuda()).float()
sphereMask = Variable(torch.from_numpy(sphereMask).cuda()).float()

# define network forward for coarse model
network_coarse = networkForward_basic(coarseModel, shadingLayer_coarse, 64, 1)
network_coarse.model.cuda()
network_fine = networkForward_basic(fineModel, shadingLayer, args.imageSize, 2)
network_fine.model.cuda()

# tran loader for IIW evaluation
trainLoaderHelper_IIW = constructDataLoader('train', 'compare', args.imageSize)
trainLoader_IIW = torch.utils.data.DataLoader(trainLoaderHelper_IIW, 
	batch_size=args.batchSize, shuffle=True, num_workers=5)

valLoaderHelper_IIW = constructDataLoader('val', 'compare', args.imageSize)
valLoader_IIW = torch.utils.data.DataLoader(valLoaderHelper_IIW, 
	batch_size=args.batchSize, shuffle=False, num_workers=5)

# tran loader for SAW evaluation
# NOTE: the biggest difference is data augmentation
# For comparison, we cannot rotate the image, for SAW, we could rotate 
# the image
trainLoaderHelper_SAW = constructDataLoader('train', 'shading', args.imageSize)
trainLoader_SAW = torch.utils.data.DataLoader(trainLoaderHelper_SAW,
	batch_size=args.batchSize, shuffle=True, num_workers=5)

valLoaderHelper_SAW = constructDataLoader('val', 'shading', args.imageSize)
valLoader_SAW = torch.utils.data.DataLoader(valLoaderHelper_SAW,
	batch_size=args.batchSize, shuffle=False, num_workers=5)

def help_computeNetwork(data_compare, data_SAW, writer, epoch, i, indType='train'):
    # return loss
    # print i
    #begin_time = time.time()
    
    image_SUNCG_compare, image_IIW_compare, _, label_SUNCG_compare, \
            label_IIW_compare, _, IIW_name = data_compare 

    images_IIW_compare = Variable(image_IIW_compare.cuda()).float()
    output_IIW_compare_coarse = network_coarse.forward(images_IIW_compare) 
    output_IIW_compare = network_fine.forward(images_IIW_compare, output_IIW_compare_coarse) 

    return output_IIW_compare, IIW_name

labelToGPU = labelToGPU()
# prepare for training
savePath = args.savePath + '_{:0.4f}_{:0.2f}_{:04d}'.format(args.lr, args.wd, args.epochs)
if not os.path.exists(savePath):
    os.makedirs(savePath)
saveIntermedia = os.path.join(savePath, 'trained_model')
if not os.path.exists(saveIntermedia):
    os.makedirs(saveIntermedia)
saveImages = os.path.join(savePath, 'images')
if not os.path.exists(saveImages):
    os.makedirs(saveImages)
writer = SummaryWriter(os.path.join(savePath, 'tensorboard'))
optimizer = optim.Adam(fineModel.parameters(), lr = args.lr, weight_decay=args.wd)

def main():
    begin_time = time.time()
    print 'learning rate is %.6f' % args.lr
    print 'weight decay is %.6f' % args.wd
    print 'epoch is %05d' % args.epochs

    for epoch in range(args.epochs):
        train_SAW_iter = iter(trainLoader_SAW)
        val_SAW_iter = iter(valLoader_SAW)
        network_fine.model.train(False)
        for (i, data_compare) in enumerate(trainLoader_IIW, 0): 
            optimizer.zero_grad()
            data_SAW = next(train_SAW_iter, None)
            output, IIW_name = help_computeNetwork(data_compare, data_SAW, writer, epoch, i, indType='train')
            for (i, fileName) in enumerate(IIW_name):
                albedo = output['albedo'][i].data.cpu().numpy()
                normal = output['normal'][i].data.cpu().numpy()
                shading = output['shading'][i].data.cpu().numpy()
                lighting = output['lighting'][i].data.cpu().numpy()
                np.save(os.path.join(saveImages, fileName + '_albedo.npy'), albedo)
                np.save(os.path.join(saveImages, fileName + '_shading.npy'), shading)
                np.save(os.path.join(saveImages, fileName + '_normal.npy'), normal)
                np.save(os.path.join(saveImages, fileName + '_lighting.npy'), lighting)

        network_fine.model.train(False)

        for (i, data_compare) in enumerate(valLoader_IIW, 0): 
            data_SAW = next(val_SAW_iter, None)
            output, IIW_name = help_computeNetwork(data_compare, data_SAW, writer, epoch, i, indType='test')
            for (i, fileName) in enumerate(IIW_name):
                albedo = output['albedo'][i].data.cpu().numpy()
                normal = output['normal'][i].data.cpu().numpy()
                shading = output['shading'][i].data.cpu().numpy()
                lighting = output['lighting'][i].data.cpu().numpy()
                np.save(os.path.join(saveImages, fileName + '_albedo.npy'), albedo)
                np.save(os.path.join(saveImages, fileName + '_shading.npy'), shading)
                np.save(os.path.join(saveImages, fileName + '_normal.npy'), normal)
                np.save(os.path.join(saveImages, fileName + '_lighting.npy'), lighting)

if __name__ =='__main__':
    main()
