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
    fileListName=os.path.join(baseFolder, 'testing.list'),
    transform = transformer_test)

testLoader = torch.utils.data.DataLoader(testLoaderHelper, 
    batch_size=20, shuffle=False, num_workers=5)

def getImage(image, output_albedo, output_shading, output_sphere, output_normal, count, resultPath):
    '''
        compute the loss for the mini-batch
    '''
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    
    for i in range(output_albedo.size()[0]):
        imgAlbedo = np.squeeze(output_albedo[i].cpu().data.numpy())
        imgShading = np.squeeze(output_shading[i].cpu().data.numpy())
        imgSphere = np.squeeze(output_sphere[i].cpu().data.numpy())
        imgNormal = np.squeeze(output_normal[i].cpu().data.numpy())
        
        imgAlbedo = imgAlbedo.transpose((1,2,0))
        imgShading = imgShading.transpose((1,2,0))
        imgSphere = imgSphere.transpose((1,2,0))
        imgNormal = imgNormal.transpose((1,2,0))
        imgNormal = imgNormal / (np.linalg.norm(imgNormal, axis=-1, keepdims=True) + 1e-6)
        
        imgAlbedo = ((imgAlbedo-np.min(imgAlbedo))/(np.max(imgAlbedo) - np.min(imgAlbedo))*255).astype(np.uint8)
        
        ind = imgShading > 1
        imgShading[ind] = 1
        ind = imgShading < 0
        imgShading[ind] = 0
        imgShading = (imgShading*255).astype(np.uint8)
        
        ind = imgSphere > 1
        imgSphere[ind] = 1
        ind = imgSphere < 0
        imgSphere[ind] = 0
        imgSphere = (imgSphere*255).astype(np.uint8)
        
        imgNormal = ((imgNormal/2+0.5)*255).astype(np.uint8)
        
        io.imsave(os.path.join(resultPath, 'shading_{:04d}_{:02d}.png'.format(count, i)), imgShading)
        io.imsave(os.path.join(resultPath, 'albedo_{:04d}_{:02d}.png'.format(count, i)), imgAlbedo)
        io.imsave(os.path.join(resultPath, 'sphere_{:04d}_{:02d}.png'.format(count, i)), imgSphere)
        io.imsave(os.path.join(resultPath, 'normal_{:04d}_{:02d}.png'.format(count, i)), imgNormal)

def getLoss(image, output_albedo, output_shading, output_normal, albedo, shading, normal, mask):
    '''
        compute the loss for the mini-batch
    '''
    output_albedo_grad = gradientLayer.forward(output_albedo)
    output_shading_grad = gradientLayer.forward(output_shading)
    albedo_grad = gradientLayer.forward(albedo)
    shading_grad = gradientLayer.forward(shading)
    
    output_normal_grad = gradientLayer.forward(output_normal)
    normal_grad = gradientLayer.forward(normal)
    
    imgMask = mask.expand(-1, 3, -1, -1)
    gradMask = mask.expand(-1, 6, -1, -1)
    
    loss_albedo = criterion(imgMask, torch.abs(albedo - output_albedo))
    loss_shading = criterion(imgMask, torch.abs(shading - output_shading))
    
    loss_albedo_grad = criterion(gradMask, torch.abs(albedo_grad - output_albedo_grad))
    loss_shading_grad = criterion(gradMask, torch.abs(shading_grad - output_shading_grad))
    
    loss_normal = criterion(imgMask, torch.abs(output_normal - normal))
    loss_normal_grad = criterion(gradMask, torch.abs(normal_grad - output_normal_grad))
    
    return loss_albedo, loss_shading, loss_albedo_grad, loss_shading_grad, loss_normal, loss_normal_grad


def main(savePath, modelPath):
    begin_time = time.time()
    my_network = torch.load(os.path.join(modelPath, 'trained_model.t7'))
    my_network.cuda()
    my_network.train(False)
    test_loss = 0
    count = 0
    
    sphereNormal, _, _ = getSHNormal(64)
    # img_normal = np.transpose(sphereNormal, (0, 1,2,0))
    # img_normal = ((img_normal/2.0+0.5)*255).astype(np.uint8)
    # io.imsave('sphere_normal.png', img_normal)
    
    sphereNormal = Variable(torch.Tensor(sphereNormal).cuda(), volatile=True).float()
    tl_albedo = 0 
    tl_shading = 0
    tl_albedo_grad = 0
    tl_shading_grad = 0
    tl_normal = 0
    tl_normal_grad = 0
    for i, data in enumerate(testLoader, 0):
        inputs, albedo, shading, normal, mask = data
        inputs, albedo, shading, normal, mask = Variable(inputs.cuda(), volatile=True).float(), \
            Variable(albedo.cuda(), volatile=True).float(), \
            Variable(shading.cuda(), volatile=True).float(), \
            Variable(normal.cuda(), volatile=True).float(), \
            Variable(mask.cuda(), volatile=True).float()
        output_albedo, output_normal, output_lighting = my_network(inputs)
        output_shading = shadingLayer(output_normal, output_lighting)

        numSamples = inputs.size()[0]

        # prepare sphere
        out_sphereNormal = sphereNormal.expand(numSamples, 3, 64, 64)
        output_sphere = shadingLayer(out_sphereNormal, output_lighting)
        
        
        # let's only record 25 batches, 500 images
        if i < 25:
            getImage(inputs, output_albedo, output_shading, output_sphere, output_normal, count, savePath)

        loss_albedo, loss_shading, loss_albedo_grad, loss_shading_grad, loss_normal, loss_normal_grad = \
            getLoss(inputs, output_albedo, output_shading, output_normal, albedo, shading, normal, mask)
        
        
        tl_albedo += loss_albedo
        tl_shading += loss_shading
        tl_albedo_grad += loss_albedo_grad
        tl_shading_grad += loss_shading_grad
        tl_normal += loss_normal
        tl_normal_grad += loss_normal_grad
        
        count += 1
    print tl_albedo.data.cpu().numpy()
    print type(tl_albedo.data.cpu().numpy())
    print count
    print type(count)
    print 'albedo loss is %.4f' % (tl_albedo.data.cpu().numpy()/count)
    print 'albedo gradient loss is %.4f' % (tl_albedo_grad.data.cpu().numpy()/count)
    print 'shading loss is %.4f' % (tl_shading.data.cpu().numpy()/count)
    print 'shading gradient loss is %.4f' % (tl_shading_grad.data.cpu().numpy()/count)
    print 'normal loss is %.4f' % (tl_normal.data.cpu().numpy()/count)
    print 'normal gradient loss is %.4f' % (tl_normal_grad.data.cpu().numpy()/count)

if __name__ == '__main__':
    savePath = sys.argv[1]
    modelPath = sys.argv[2]
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    main(savePath, modelPath)
