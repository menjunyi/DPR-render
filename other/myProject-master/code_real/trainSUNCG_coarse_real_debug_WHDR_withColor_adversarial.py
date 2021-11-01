import sys
sys.path.append('../utils')
sys.path.append('../model')
sys.path.append('model_real')
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
netD_albedo_coarse, netD_albedo_fine, netD_albedo_detail = loadDiscriminator(args)

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
from defineLoss_WHDR_symmetric import *
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
network = networkForward_basic(coarseModel, shadingLayer_coarse, args.imageSize, 1)
network.model.cuda()

# everything into cuda
netD_albedo_coarse.cuda()

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

real_label = 1
fake_label = 0
criterion = nn.BCELoss()
# we stopped here
def help_computeNetwork(data_compare, data_SAW, writer, epoch, i, indType='train'):
    
    return_loss = OrderedDict()
    image_SUNCG_compare, image_IIW_compare, _, label_SUNCG_compare, \
            label_IIW_compare, _ = data_compare 

    # -------------------------------------------------------------------------
    # train on compare 
    if indType == 'train':
        Testing = False
    elif indType == 'test':
        Testing = True

    # get data
    images_SUNCG_compare = Variable(image_SUNCG_compare.cuda(), volatile=Testing).float()
    label_SUNCG_compare = labelToGPU.toGPU(label_SUNCG_compare, Testing) 
    images_IIW_compare = Variable(image_IIW_compare.cuda(), volatile=Testing).float()
    label_IIW_compare = labelToGPU.toGPU(label_IIW_compare, Testing) 

    # train on D
    numData_compare = images_SUNCG_compare.shape[0]
    numData_IIW = images_IIW_compare.shape[0]
    # real label
    label = Variable((real_label*torch.ones(numData_compare)).cuda(), volatile=Testing).float()
    output = netD_albedo_coarse(label_SUNCG_compare['albedo'])
    errD_real = criterion(output, label)
    #D_x = output.mean().item()

    # train on G
    output_SUNCG_compare = network.forward(images_SUNCG_compare) 
    loss_SUNCG_compare = evaluateNetwork.IIW_loss(output_SUNCG_compare, label_SUNCG_compare, 'SUNCG', scale_factor=4, trainType=indType)
    
    output_IIW_compare = network.forward(images_IIW_compare) 
    loss_IIW_compare = evaluateNetwork.IIW_loss(output_IIW_compare, label_IIW_compare, 'IIW', scale_factor=4, trainType=indType)
    # compute color loss and reconstruction loss
    loss_IIW_color = evaluateNetwork.IIW_colorLoss(output_IIW_compare, images_IIW_compare, label_IIW_compare)

    # fake label
    label = Variable((fake_label*torch.ones(numData_IIW)).cuda(), volatile=Testing).float()
    # cut the connection of albedo and network that output albedo
    detached_albedo = Variable(output_IIW_compare['albedo'].data)
    output = netD_albedo_coarse(detached_albedo)
    errD_fake = criterion(output, label)
    errD = (errD_real + errD_fake)/2.0
    if indType == 'train':
        errD.backward()
        optimizer_D_albedo.step()
    return_loss['D_loss_real'] = errD_real
    return_loss['D_loss_fake'] = errD_fake

    # loss = 50*(loss_SUNCG_compare['WHDR'] + loss_IIW_compare['WHDR'])
    # loss = 10*loss_IIW_compare['WHDR']
    #loss = 10*loss_IIW_compare['WHDR'] + loss_IIW_color['color'] + loss_IIW_color['color_grad'] + \
    #        loss_IIW_color['reconstruct'] + loss_IIW_color['reconstruct_grad']
    label = Variable((real_label*torch.ones(numData_IIW)).cuda()).float()
    output = netD_albedo_coarse(output_IIW_compare['albedo'])
    errG = criterion(output, label)
    

    # combine all losses
    loss = 10*loss_IIW_compare['WHDR'] + loss_IIW_color['color']  + 5*errG

    return_loss['G_loss'] = errG
    return_loss['SUNCG_WHDR'] = loss_SUNCG_compare['WHDR'].data[0]
    return_loss['IIW_WHDR'] = loss_IIW_compare['WHDR'].data[0]
    return_loss['IIW_color'] = loss_IIW_color['color'].data[0]
    return_loss['IIW_color_grad'] = loss_IIW_color['color_grad'].data[0]
    return_loss['IIW_reconstruct'] = loss_IIW_color['reconstruct'].data[0]
    return_loss['IIW_reconstruct_grad'] = loss_IIW_color['reconstruct_grad'].data[0]

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
    images_SUNCG_SAW = Variable(image_SUNCG_SAW.cuda(), volatile=Testing).float()
    output_SUNCG_SAW = network.forward(images_SUNCG_SAW) 
    label_SUNCG_SAW = labelToGPU.toGPU(label_SUNCG_SAW, Testing) 
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
        #loss.backward()
        optimizer.step()
    #print 'time used for others %s' % (time.time() - begin_time)

    # record images
    if i % args.print_freq == args.print_freq - 1:
        writer.add_image(os.path.join(indType, 'SUNCG_albedo'), output_SUNCG_compare['albedo'])
        writer.add_image(os.path.join(indType, 'SUNCG_shading'), output_SUNCG_compare['shading'])
        writer.add_image(os.path.join(indType, 'SUNCG_normal'), output_SUNCG_compare['normal'])
        writer.add_image(os.path.join(indType, 'IIW_albedo'), output_IIW_compare['albedo'])
        writer.add_image(os.path.join(indType, 'IIW_shading'), output_IIW_compare['shading'])
        writer.add_image(os.path.join(indType, 'IIW_normal'), output_IIW_compare['normal'])
        #writer.add_image(os.path.join(indType, 'IIW_albedo'), output_IIW_SAW['albedo'])
        #writer.add_image(os.path.join(indType, 'IIW_shading'), output_IIW_SAW['shading'])
        #writer.add_image(os.path.join(indType, 'IIW_normal'), output_IIW_SAW['normal'])
        #writer.add_image(os.path.join(indType, 'NYU_albedo'), output_NYU_SAW['albedo'])
        #writer.add_image(os.path.join(indType, 'NYU_shading'), output_NYU_SAW['shading'])
        #writer.add_image(os.path.join(indType, 'NYU_normal'), output_NYU_SAW['normal'])

        writer.add_image(os.path.join(indType, 'SUNCG_normal_true'), label_SUNCG_compare['normal'])
        writer.add_image(os.path.join(indType, 'SUNCG_shading_true'), label_SUNCG_compare['shading'])
        writer.add_image(os.path.join(indType, 'SUNCG_albedo_true'), label_SUNCG_compare['albedo'])

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
optimizer = optim.Adam(coarseModel.parameters(), lr = args.lr, weight_decay=args.wd)
optimizer_D_albedo = optim.Adam(netD_albedo_coarse.parameters(), lr = args.lr, weight_decay=args.wd)

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
    lossName = ['D_loss_real', 'D_loss_fake', 'G_loss', 'SUNCG_WHDR', 'IIW_WHDR', 'IIW_color', 'IIW_color_grad', 'IIW_reconstruct', 
            'IIW_reconstruct_grad', 'SUNCG_albedo_loss', 'SUNCG_shading_loss', 
            'SUNCG_normal_loss', 'SUNCG_lighting']
    print>>fid_sep, ','.join(lossName) 
    print>>fid_test_sep, ','.join(lossName)
    numIte_training = 0
    numIte_testing = 0

    for epoch in range(args.epochs):
        train_SAW_iter = iter(trainLoader_SAW)
        val_SAW_iter = iter(valLoader_SAW)
        begin_time_inner = time.time()
        network.model.train(True)
        loss_list = []
        tmp_loss_list = []
        for (i, data_compare) in enumerate(trainLoader_IIW, 0): 
            optimizer.zero_grad()
            optimizer_D_albedo.zero_grad()
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
    network.model.cpu()
    torch.save(network.model, os.path.join(savePath, 'trained_model.t7'))
    fid.close()
    fid_test.close()
    fid_sep.close()
    fid_test_sep.close()

if __name__ =='__main__':
    main()
