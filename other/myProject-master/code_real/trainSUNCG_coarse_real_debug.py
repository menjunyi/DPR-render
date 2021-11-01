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
print type(coarseModel)

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
from loadData_basic import *

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
criterion_IIW_paper = WhdrTestLoss_Paper()
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
print type(network.model)
network.model.cuda()

baseFolder = '/scratch1/intrinsicImage/synthetic_SUNCG/'
transformer = transforms.Compose([cropImg(output_size=64), ToTensor()])
transformer_test = transforms.Compose([testTransfer(), ToTensor()])

trainLoaderHelper = SUNCG(dataFolder = os.path.join(baseFolder, 'images_color'), 
	albedoFolder=os.path.join(baseFolder, 'albedo'),
	shadingFolder=os.path.join(baseFolder, 'shading'),
	normalFolder='/scratch1/data/SUNCG_groundTruth/SUNCG_normal',
	maskFolder=os.path.join(baseFolder, 'images_mask_correct'),
	fileListName=os.path.join(baseFolder, 'training.list'),
	transform = transformer)

testLoaderHelper = SUNCG(dataFolder=os.path.join(baseFolder, 'images_color'),
	albedoFolder=os.path.join(baseFolder, 'albedo'),
	shadingFolder=os.path.join(baseFolder, 'shading'),
	normalFolder='/scratch1/data/SUNCG_groundTruth/SUNCG_normal',
	maskFolder=os.path.join(baseFolder, 'images_mask_correct'),
	fileListName=os.path.join(baseFolder, 'testing.list'),
	transform = transformer_test)

trainLoader = torch.utils.data.DataLoader(trainLoaderHelper, 
	batch_size=20, shuffle=True, num_workers=5)

testLoader = torch.utils.data.DataLoader(testLoaderHelper, 
	batch_size=20, shuffle=False, num_workers=5)

##--------------------------------------------------------------------------------------

def help_computeNetwork(data_compare, data_SAW, writer, epoch, i, indType='train'):
    
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

    image_SUNCG_SAW,label_SUNCG_SAW = data_compare
    # suncg SAW 
    images_SUNCG_SAW = Variable(image_SUNCG_SAW.cuda()).float()
    output_SUNCG_SAW = network.forward(images_SUNCG_SAW) 
    label_SUNCG_SAW = labelToGPU.toGPU(label_SUNCG_SAW) 
    loss_SUNCG_SAW = evaluateNetwork.SUNCG_data(output_SUNCG_SAW, label_SUNCG_SAW)

    loss_light_pos = nonNegativeLighting_coarseLevel(samplingLightLayer, sphereDirection_pos, output_SUNCG_SAW['lighting'])
    loss_light_neg = nonNegativeLighting_coarseLevel(samplingLightLayer, sphereDirection_neg, output_SUNCG_SAW['lighting'])
    loss_light = loss_light_neg + loss_light_pos

    return_loss['SUNCG_albedo'] = loss_SUNCG_SAW['albedo'].data[0]
    return_loss['SUNCG_shading'] = loss_SUNCG_SAW['shading'].data[0]
    return_loss['SUNCG_normal'] = loss_SUNCG_SAW['normal'].data[0]
    return_loss['SUNCG_lighting'] = loss_light.data[0]
    
    loss = loss_SUNCG_SAW['albedo'] + loss_SUNCG_SAW['normal'] + loss_SUNCG_SAW['shading'] \
            + loss_SUNCG_SAW['albedo_grad'] + loss_SUNCG_SAW['normal_grad'] + loss_SUNCG_SAW['shading_grad'] + loss_light


    if indType=='train':
        loss.backward()
        optimizer.step()
    #print 'time used for others %s' % (time.time() - begin_time)

    # record images
    if i % args.print_freq == args.print_freq - 1:
        writer.add_image(os.path.join(indType, 'SUNCG_albedo'), output_SUNCG_SAW['albedo'])
        writer.add_image(os.path.join(indType, 'SUNCG_shading'), output_SUNCG_SAW['shading'])
        writer.add_image(os.path.join(indType, 'SUNCG_normal'), output_SUNCG_SAW['normal'])

        writer.add_image(os.path.join(indType, 'SUNCG_normal_true'), label_SUNCG_SAW['normal'])
        writer.add_image(os.path.join(indType, 'SUNCG_shading_true'), label_SUNCG_SAW['shading'])
        writer.add_image(os.path.join(indType, 'SUNCG_albedo_true'), label_SUNCG_SAW['albedo'])

    return return_loss

def record_loss(writer, loss_list, lossName, numIte, indType='train'):
    '''
        write loss into tensorboard
    '''
    for (i, item) in enumerate(lossName):
        writer.add_scalar(os.path.join(indType, item), loss_list[i], numIte)


labelToGPU = labelToGPU()
# prepare for training
savePath = args.savePath + '_{:0.4f}_{:0.2f}_{:04d}'.format(args.lr, args.wd, args.epochs)
if not os.path.exists(savePath):
    os.makedirs(savePath)
saveIntermedia = os.path.join(savePath, 'trained_model')
if not os.path.exists(saveIntermedia):
    os.makedirs(saveIntermedia)
writer = SummaryWriter(os.path.join(savePath, 'tensorboard'))
optimizer = optim.Adam(network.model.parameters(), lr = args.lr, weight_decay=args.wd)

def main():
    begin_time = time.time()
    print 'learning rate is %.6f' % args.lr
    print 'weight decay is %.6f' % args.wd
    print 'epoch is %05d' % args.epochs
    fid = open(os.path.join(savePath, 'training.log'), 'w')
    fid_sep = open(os.path.join(savePath, 'training_sep.log'), 'w')
    fid_test = open(os.path.join(savePath, 'testing.log'), 'w')
    fid_test_sep = open(os.path.join(savePath, 'testing_sep.log'), 'w')
    lossName = ['SUNCG_albedo_loss', 'SUNCG_shading_loss', 'SUNCG_normal_loss']
    print>>fid_sep, ','.join(lossName) 
    print>>fid_test_sep, ','.join(lossName)
    numIte_training = 0
    numIte_testing = 0
    for epoch in range(args.epochs):
        begin_time_inner = time.time()
        network.model.train(True)
        loss_list = []
        tmp_loss_list = []
        for (i, data) in enumerate(trainLoader, 0): 
            optimizer.zero_grad()
            #data_SAW = next(train_SAW_iter, None)
            inputs, albedo, shading, normal, mask, fileName = data
            data_compare = []
            data_compare.append(inputs)
            tmp = {}
            tmp['albedo'] = albedo
            tmp['shading'] = shading
            tmp['normal'] = normal
            tmp['mask'] = mask
            data_compare.append(tmp)
            return_loss = help_computeNetwork(data_compare, data_compare, writer, epoch, i, indType='train')
            numLoss = len(return_loss.values())
            loss_list.append(return_loss.values())
            tmp_loss_list.append(return_loss.values())
            if i % args.print_freq == args.print_freq-1:
                print numLoss
                numIte_training = numIte_training+1
                tmp_loss_list = np.array(tmp_loss_list)
                tmp_loss_list = np.mean(tmp_loss_list, axis=0)
                print '[%d %5d] loss: ' % (epoch + 1, i+1),
                print '%.4f '*numLoss % tuple(tmp_loss_list)
                print>>fid, '%d %5d ' % (epoch+1, i+1),
                print>>fid, '%.4f '*numLoss % tuple(tmp_loss_list)
                record_loss(writer, tmp_loss_list, lossName, numIte_training, indType='train')
                tmp_loss_list = []
                print 'time for 20 iterations is %s' % (time.time() - begin_time_inner)
                begin_time_inner = time.time()
        loss_list = np.mean(np.array(loss_list), axis=0)
        print >> fid_sep, '%0.6f '* numLoss % tuple(loss_list)
        print '%0.6f '* numLoss % tuple(loss_list)

        network.model.train(False)
        test_loss_list = []

        for (i, data_compare) in enumerate(valLoader_IIW, 0): 
            data_SAW = next(val_SAW_iter, None)
            return_loss = help_computeNetwork(data_compare, data_SAW, writer, epoch, i, indType='test')
            numLoss = len(return_loss.values())
            test_loss_list.append(return_loss.values())
        test_loss_list = np.array(test_loss_list)
        test_loss_list = np.mean(test_loss_list, axis=0)
        print '[%d %5d] loss: ' % (epoch + 1, i+1),
        print '%.4f '*numLoss % tuple(test_loss_list)
        print>>fid, '%d %5d ' % (epoch+1, i+1),
        print>>fid, '%.4f '*numLoss % tuple(test_loss_list)
        record_loss(writer, test_loss_list, lossName, numIte_training, indType='test')
        print >> fid_test_sep, '%0.6f '* numLoss % tuple(test_loss_list)
        print '%0.6f '* numLoss % tuple(test_loss_list)
        tmp_saveName = os.path.join(saveIntermedia, 'trained_model_{:02d}.t7'.format(epoch))
        network.model.cpu()
        torch.save(network.model.state_dict(), tmp_saveName)
        network.model.cuda()
    fid.close()
    fid_test.close()
    fid_sep.close()
    fid_test_sep.close()

if __name__ =='__main__':
    main()
