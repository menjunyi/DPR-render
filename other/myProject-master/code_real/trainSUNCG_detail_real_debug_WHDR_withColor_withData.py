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
#network_coarse = networkForward_basic(coarseModel, shadingLayer_coarse, 64, 1)
#network_coarse.model.cuda()
#network_fine = networkForward_basic(fineModel, shadingLayer, 128, 2)
#network_fine.model.cuda()
network_detail = networkForward_basic(detailModel, shadingLayer, args.imageSize, 3)
network_detail.model.cuda()

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
    
    return_loss = OrderedDict()
    image_SUNCG_compare, image_IIW_compare, _, label_SUNCG_compare, \
            label_IIW_compare, _, IIW_name = data_compare 
    # train on compare 
    #images_SUNCG_compare = Variable(image_SUNCG_compare.cuda()).float()
    #output_SUNCG_compare_coarse = network_coarse.forward(images_SUNCG_compare) 
    #output_SUNCG_compare_fine = network_fine.forward(images_SUNCG_compare, output_SUNCG_compare_coarse) 
    #output_SUNCG_compare = network_detail.forward(images_SUNCG_compare, output_SUNCG_compare_fine) 
    #label_SUNCG_compare = labelToGPU.toGPU(label_SUNCG_compare, indType) 
    #loss_SUNCG_compare = evaluateNetwork.IIW_loss(output_SUNCG_compare, label_SUNCG_compare, 'SUNCG', scale_factor=2, trainType=indType)
    ##loss_SUNCG_compare = evaluateNetwork.IIW_loss(output_IIW_compare, label_SUNCG_compare, 'SUNCG', scale_factor=4, trainType=indType)
    if indType == 'train':
        Testing = False
    elif indType == 'test':
        Testing = True
    
    images_IIW_compare = Variable(image_IIW_compare.cuda()).float()
    #print label_IIW_compare
    label_IIW_compare = labelToGPU.toGPU(label_IIW_compare, Testing) 
    #output_IIW_compare_coarse = network_coarse.forward(images_IIW_compare) 
    #output_IIW_compare_fine = network_fine.forward(images_IIW_compare, output_IIW_compare_coarse) 
    # load data form file
    numImages = len(IIW_name)
    output_IIW_compare_fine = {}
    output_IIW_compare_fine['albedo'] = np.zeros((numImages, 3, 128, 128))
    output_IIW_compare_fine['shading'] = np.zeros((numImages, 3, 128, 128))
    output_IIW_compare_fine['normal'] = np.zeros((numImages, 3, 128, 128))
    output_IIW_compare_fine['lighting'] = np.zeros((numImages, 27, 128, 128))
    finePath_help = args.fineModel_load.split('/')
    finePath = '/'.join(finePath_help[0:-1])
    for (i, item) in enumerate(IIW_name):
        fine_albedo = np.load(os.path.join(finePath, 'images', item + '_albedo.npy'))
        fine_normal = np.load(os.path.join(finePath, 'images', item + '_normal.npy'))
        fine_shading = np.load(os.path.join(finePath, 'images', item + '_shading.npy'))
        fine_lighting= np.load(os.path.join(finePath, 'images', item + '_lighting.npy'))
        output_IIW_compare_fine['albedo'][i] = fine_albedo
        output_IIW_compare_fine['normal'][i] = fine_normal
        output_IIW_compare_fine['shading'][i] = fine_shading
        output_IIW_compare_fine['lighting'][i] = fine_lighting
    for item in output_IIW_compare_fine.keys():
        output_IIW_compare_fine[item] = Variable(torch.from_numpy(output_IIW_compare_fine[item]).cuda()).float()

    output_IIW_compare = network_detail.forward(images_IIW_compare, output_IIW_compare_fine) 
    loss_IIW_compare = evaluateNetwork.IIW_loss(output_IIW_compare, label_IIW_compare, 'IIW', scale_factor=2, trainType=indType)
    # compute color loss and reconstruction loss
    loss_IIW_color = evaluateNetwork.IIW_colorLoss(output_IIW_compare, images_IIW_compare, label_IIW_compare)

    loss = 10*loss_IIW_compare['WHDR'] + loss_IIW_color['color'] 

    return_loss['IIW_WHDR'] = loss_IIW_compare['WHDR'].data[0]
    return_loss['IIW_color'] = loss_IIW_color['color'].data[0]
    return_loss['IIW_color_grad'] = loss_IIW_color['color_grad'].data[0]
    return_loss['IIW_reconstruct'] = loss_IIW_color['reconstruct'].data[0]
    return_loss['IIW_reconstruct_grad'] = loss_IIW_color['reconstruct_grad'].data[0]
    
    # backward for IIW compare loss
    if indType=='train':
        loss.backward()


    if indType=='train':
        #loss.backward()
        optimizer.step()
    #print 'time used for others %s' % (time.time() - begin_time)

    # record images
    if i % args.print_freq == args.print_freq - 1:
        writer.add_image(os.path.join(indType, 'IIW_albedo'), output_IIW_compare['albedo'])
        writer.add_image(os.path.join(indType, 'IIW_shading'), output_IIW_compare['shading'])
        writer.add_image(os.path.join(indType, 'IIW_normal'), output_IIW_compare['normal'])
        #writer.add_image(os.path.join(indType, 'IIW_albedo'), output_IIW_SAW['albedo'])
        #writer.add_image(os.path.join(indType, 'IIW_shading'), output_IIW_SAW['shading'])
        #writer.add_image(os.path.join(indType, 'IIW_normal'), output_IIW_SAW['normal'])
        #writer.add_image(os.path.join(indType, 'NYU_albedo'), output_NYU_SAW['albedo'])
        #writer.add_image(os.path.join(indType, 'NYU_shading'), output_NYU_SAW['shading'])
        #writer.add_image(os.path.join(indType, 'NYU_normal'), output_NYU_SAW['normal'])

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
optimizer = optim.Adam(detailModel.parameters(), lr = args.lr, weight_decay=args.wd)

def main():
    begin_time = time.time()
    print 'learning rate is %.6f' % args.lr
    print 'weight decay is %.6f' % args.wd
    print 'epoch is %05d' % args.epochs
    fid = open(os.path.join(savePath, 'training.log'), 'w')
    fid_sep = open(os.path.join(savePath, 'training_sep.log'), 'w')
    fid_test = open(os.path.join(savePath, 'testing.log'), 'w')
    fid_test_sep = open(os.path.join(savePath, 'testing_sep.log'), 'w')
    #lossName = ['SUNCG_WDHR', 'IIW_WDHR', 'SUNCG_albedo', 'SUNCG_albedo_grad',
    #        'SUNCG_shading', 'SUNCG_shading_normal', 'SUNCG_normal', 'SUNCG_normal_grad',
    #        'IIW_SAW_1', 'IIW_SAW_2', 'IIW_reconstruct', 'NYU_SAW_1', 'NYU_SAW_2', 'NYU_reconstruct',
    #        'NYU_normal', 'NYU_normal_grad']
    #lossName = ['SUNCG_WDHR_loss', 'IIW_WDHR_loss', 'SUNCG_albedo_loss', 'SUNCG_shading_loss', 'SUNCG_normal_loss']
    #lossName = ['SUNCG_WHDR', 'IIW_WHDR', 'IIW_color', 'IIW_color_grad', 'IIW_reconstruct', 
    #        'IIW_reconstruct_grad', 'SUNCG_albedo_loss', 'SUNCG_shading_loss', 
    #        'SUNCG_normal_loss']
    lossName = ['IIW_WHDR', 'IIW_color', 'IIW_color_grad', 'IIW_reconstruct', 
            'IIW_reconstruct_grad']
    print>>fid_sep, ','.join(lossName) 
    print>>fid_test_sep, ','.join(lossName)
    numIte_training = 0
    numIte_testing = 0

    for epoch in range(args.epochs):
        train_SAW_iter = iter(trainLoader_SAW)
        val_SAW_iter = iter(valLoader_SAW)
        begin_time_inner = time.time()
        network_detail.model.train(True)
        loss_list = []
        tmp_loss_list = []
        for (i, data_compare) in enumerate(trainLoader_IIW, 0): 
            optimizer.zero_grad()
            data_SAW = next(train_SAW_iter, None)
            return_loss = help_computeNetwork(data_compare, data_SAW, writer, epoch, i, indType='train')
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

        network_detail.model.train(False)
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
        network_detail.model.cpu()
        torch.save(network_detail.model.state_dict(), tmp_saveName)
        network_detail.model.cuda()
    network_detail.model.cpu()
    torch.save(network_detail.model, os.path.join(savePath, 'trained_model.t7'))
    fid.close()
    fid_test.close()
    fid_sep.close()
    fid_test_sep.close()

if __name__ =='__main__':
    main()
