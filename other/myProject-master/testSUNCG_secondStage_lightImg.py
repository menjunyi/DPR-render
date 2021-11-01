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
from defineHourglass_128 import *
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

def getImage(image, output_albedo, output_shading, output_sphere, output_normal, gt_albedo, gt_shading, gt_normal, count, resultPath):
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

        gtAlbedo = np.squeeze(gt_albedo[i].cpu().data.numpy())
        gtShading = np.squeeze(gt_shading[i].cpu().data.numpy())
        gtNormal = np.squeeze(gt_normal[i].cpu().data.numpy())

        
        imgAlbedo = imgAlbedo.transpose((1,2,0))
        imgShading = imgShading.transpose((1,2,0))
        imgSphere = imgSphere.transpose((1,2,0))
        imgNormal = imgNormal.transpose((1,2,0))
        imgNormal = imgNormal / (np.linalg.norm(imgNormal, axis=-1, keepdims=True) + 1e-6)


        gtAlbedo = gtAlbedo.transpose((1,2,0))
        gtNormal = gtNormal.transpose((1,2,0))
        gtShading = gtShading.transpose((1,2,0))
        diffNormal = ((1 - np.abs(np.linalg.norm(imgNormal*gtNormal, axis=-1)))*255.0).astype(np.uint8)
        gtAlbedo = (gtAlbedo*255.0).astype(np.uint8)
        gtShading = (gtShading*255.0).astype(np.uint8)
        gtNormal = ((gtNormal/2.0 + 0.5)*255.).astype(np.uint8)
        #io.imsave(os.path.join(resultPath, 'gt_shading_{:04d}_{:02d}.png'.format(count, i)), gtShading)
        #io.imsave(os.path.join(resultPath, 'gt_albedo_{:04d}_{:02d}.png'.format(count, i)), gtAlbedo)
        #io.imsave(os.path.join(resultPath, 'gt_normal_{:04d}_{:02d}.png'.format(count, i)), gtNormal)
        
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
        io.imsave(os.path.join(resultPath, 'diff_normal_{:04d}_{:02d}.png'.format(count, i)), diffNormal)

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
    sphereNormal, _, _ = getSHNormal(64)
    
    sphereNormal = Variable(torch.Tensor(sphereNormal).cuda(), volatile=True).float()
    tl_albedo = 0 
    tl_shading = 0
    tl_albedo_grad = 0
    tl_shading_grad = 0
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
        inputs = torch.cat((inputs, coarse_albedo, coarse_normal, coarse_shading), dim=1)

        output_albedo, output_normal, output_lighting = my_network(inputs)

        true_lighting = output_lighting + coarse_lighting.expand(-1,-1, IMAGE_SIZE, IMAGE_SIZE)
        # get shading
        output_shading = shadingLayer(F.normalize(coarse_normal + output_normal, p=2, dim=1), 
                true_lighting)
        
        output_albedo = output_albedo + coarse_albedo
        output_normal = F.normalize((output_normal + coarse_normal), p=1, dim=1)

        # visualize the light
        size_light = 64
        numSamples = inputs.size()[0]
        # compute the average of every 16x16 pixels for lighting
        subArea = 8 
        numH = IMAGE_SIZE/subArea
        numW = IMAGE_SIZE/subArea
        whole_sphere = Variable(torch.zeros(numSamples, 3, numH*size_light, numW*size_light).cuda()).float()
        out_sphereNormal = sphereNormal.expand(numSamples, 3, size_light, size_light)
        for i in range(numH):
            for j in range(numW):
                tmpLight = true_lighting[:,:,i*subArea:(i+1)*subArea, j*subArea: (j+1)*subArea]
                tmpLight = torch.mean(torch.mean(tmpLight, dim=-1), dim=-1)
                output_sphere = shadingLayer_coarse(out_sphereNormal, tmpLight)
                whole_sphere[:,:,i*size_light:(i+1)*size_light, j*size_light : (j+1)*size_light] = output_sphere
        
        # let's only record 25 batches, 500 images
        if ii < 25:
            getImage(inputs, output_albedo, output_shading, whole_sphere, 
                    output_normal, albedo, shading, normal, count, savePath)
        else:
            break

        loss_albedo, loss_shading, loss_albedo_grad, loss_shading_grad, loss_normal, loss_normal_grad = \
            getLoss(inputs, output_albedo, output_shading, output_normal, albedo, shading, normal, mask)
        
        
        tl_albedo += loss_albedo
        tl_shading += loss_shading
        tl_albedo_grad += loss_albedo_grad
        tl_shading_grad += loss_shading_grad
        tl_normal += loss_normal
        tl_normal_grad += loss_normal_grad
        
        count += 1
    #print tl_albedo.data.cpu().numpy()
    #print type(tl_albedo.data.cpu().numpy())
    #print count
    #print type(count)
    #print 'albedo loss is %.4f' % (tl_albedo.data.cpu().numpy()/count)
    #print 'albedo gradient loss is %.4f' % (tl_albedo_grad.data.cpu().numpy()/count)
    #print 'shading loss is %.4f' % (tl_shading.data.cpu().numpy()/count)
    #print 'shading gradient loss is %.4f' % (tl_shading_grad.data.cpu().numpy()/count)
    #print 'normal loss is %.4f' % (tl_normal.data.cpu().numpy()/count)
    #print 'normal gradient loss is %.4f' % (tl_normal_grad.data.cpu().numpy()/count)

if __name__ == '__main__':
    savePath = sys.argv[1]
    coarseModelPath = sys.argv[2]
    modelPath = sys.argv[3]
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    main(savePath, coarseModelPath, modelPath)
