import sys
sys.path.append('utils')
sys.path.append('model')

import os
import numpy as np
import time

from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F

# load module defined by myself
from defineNormal import *
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

IMAGE_SIZE=128

gradientLayer = gradientLayer_color()
gradientLayer.cuda()
shadingLayer_coarse = constructShading()
shadingLayer_coarse.cuda()
shadingLayer = constructShading_lightImg()
shadingLayer.cuda()
samplingLightLayer = samplingLight()
samplingLightLayer.cuda()

transformer_test = transforms.Compose([testTransfer(output_size=IMAGE_SIZE), ToTensor()])
testLoaderHelper = SUNCG(dataFolder=os.path.join(baseFolder, 'images_color'),
    albedoFolder=os.path.join(baseFolder, 'albedo'),
    shadingFolder=os.path.join(baseFolder, 'shading'),
    normalFolder='/scratch1/data/SUNCG_groundTruth/SUNCG_normal',
    maskFolder=os.path.join(baseFolder, 'images_mask_correct'),
    fileListName=os.path.join(baseFolder, 'testing.list'),
    transform = transformer_test)
testLoader = torch.utils.data.DataLoader(testLoaderHelper, 
    batch_size=20, shuffle=False, num_workers=5)

def getImage(output_normal, gt_normal, count, resultPath):
    '''
        compute the loss for the mini-batch
    '''
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    
    for i in range(output_normal.size()[0]):
        imgNormal = np.squeeze(output_normal[i].cpu().data.numpy())
        gtNormal = np.squeeze(gt_normal[i].cpu().data.numpy())
        
        imgNormal = imgNormal.transpose((1,2,0))
        imgNormal = imgNormal / (np.linalg.norm(imgNormal, axis=-1, keepdims=True) + 1e-6)


        gtNormal = gtNormal.transpose((1,2,0))
        diffNormal = ((1 - np.abs(np.linalg.norm(imgNormal*gtNormal, axis=-1)))*255.0).astype(np.uint8)
        gtNormal = ((gtNormal/2.0 + 0.5)*255.).astype(np.uint8)
        
        
        imgNormal = ((imgNormal/2+0.5)*255).astype(np.uint8)
        
        io.imsave(os.path.join(resultPath, 'normal_{:04d}_{:02d}.png'.format(count, i)), imgNormal)
        io.imsave(os.path.join(resultPath, 'diff_normal_{:04d}_{:02d}.png'.format(count, i)), diffNormal)

def getLoss(output_normal, normal, mask):
    '''
        compute the loss for the mini-batch
    '''
    output_normal_grad = gradientLayer.forward(output_normal)
    normal_grad = gradientLayer.forward(normal)
    
    imgMask = mask.expand(-1, 3, -1, -1)
    gradMask = mask.expand(-1, 6, -1, -1)
    
    loss_normal = criterion(imgMask, torch.abs(output_normal - normal))
    loss_normal_grad = criterion(gradMask, torch.abs(normal_grad - output_normal_grad))
    
    return loss_normal, loss_normal_grad


def main(savePath, coarseModelPath, modelPath):
    begin_time = time.time()

    my_network = torch.load(os.path.join(modelPath, 'trained_model.t7'))
    my_network.cuda()
    my_network.train(False)

    global my_network_coarse 
    my_network_coarse = torch.load(os.path.join(coarseModelPath, 'trained_model.t7'))
    my_network_coarse.cuda()
    my_network_coarse.train(False)

    test_loss = 0
    count = 0
    
    tl_normal = 0
    tl_normal_grad = 0
    for ii, data in enumerate(testLoader, 0):
        print 'start...'
        inputs, albedo, shading, normal, mask, fileName = data
        inputs, albedo, shading, normal, mask = Variable(inputs.cuda(), volatile=True).float(), \
            Variable(albedo.cuda(), volatile=True).float(), \
            Variable(shading.cuda(), volatile=True).float(), \
            Variable(normal.cuda(), volatile=True).float(), \
            Variable(mask.cuda(), volatile=True).float()
        # --------------------------------------------------------------------------------
        # get albedo, shading, normal, lighting in coarse scale and prepare the residual
        coarse_albedo, coarse_normal, coarse_lighting = \
                my_network_coarse(F.upsample(inputs, size=[64, 64], mode='bilinear'))
        coarse_shading = shadingLayer_coarse(coarse_normal, coarse_lighting)
        coarse_albedo = F.upsample(coarse_albedo, size=[IMAGE_SIZE, IMAGE_SIZE], mode='bilinear')
        coarse_normal = F.upsample(coarse_normal, size=[IMAGE_SIZE, IMAGE_SIZE], mode='bilinear')
        coarse_shading = F.upsample(coarse_shading, size=[IMAGE_SIZE, IMAGE_SIZE], mode='bilinear')

        # NOTE: we have a bug in coarse network for lighting, correct it
        coarse_lighting = Variable(coarse_lighting[:,0:27].data).float()
        coarse_lighting = coarse_lighting.unsqueeze(-1)
        coarse_lighting = coarse_lighting.unsqueeze(-1)

        # concatenate images, albedo, normal, shading as input
        inputs_normal = torch.cat((inputs, coarse_normal), dim=1)
        diff_normal = my_network(inputs_normal)
        output_normal = F.normalize(diff_normal + coarse_normal, p=2, dim=1)

        # let's only record 25 batches, 500 images
        if ii < 25:
            getImage(output_normal, normal, count, savePath)
        else:
            break

        loss_normal, loss_normal_grad = getLoss(output_normal, normal, mask)
        
        
        tl_normal += loss_normal
        tl_normal_grad += loss_normal_grad
        
        count += 1
    #print tl_albedo.data.cpu().numpy()
    #print type(tl_albedo.data.cpu().numpy())
    #print count
    #print type(count)
    print 'normal loss is %.4f' % (tl_normal.data.cpu().numpy()/count)
    print 'normal gradient loss is %.4f' % (tl_normal_grad.data.cpu().numpy()/count)

if __name__ == '__main__':
    savePath = sys.argv[1]
    coarseModelPath = sys.argv[2]
    modelPath = sys.argv[3]
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    main(savePath, coarseModelPath, modelPath)
