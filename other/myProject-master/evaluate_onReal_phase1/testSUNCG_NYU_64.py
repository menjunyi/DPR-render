import sys
sys.path.append('../utils')
sys.path.append('../model')
sys.path.append('../evaluation')

import os
import numpy as np
import time

from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F

# load module defined by myself
#from defineHourglass_128 import *
from utils_shading import *
from loadNYU import *
from loadIIW import *
from defineHelp import *
from defineHelp_lighting import *

# set random seed for all possible randomness
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


baseFolder = '/scratch1/other_code/NYUv2/dataset/'
IMAGE_SIZE=64

gradientLayer = gradientLayer_color()
gradientLayer.cuda()
shadingLayer_coarse = constructShading()
shadingLayer_coarse.cuda()
shadingLayer = constructShading_lightImg()
shadingLayer.cuda()
samplingLightLayer = samplingLight()
samplingLightLayer.cuda()

transformer_test = transforms.Compose([testTransfer(output_size=IMAGE_SIZE), ToTensor()])
testLoaderHelper = NYU(dataFolder=os.path.join(baseFolder, 'image'),
    normalFolder = os.path.join(baseFolder, 'normal'),
    maskFolder = os.path.join(baseFolder, 'normal'),
    fileListName=os.path.join(baseFolder, 'testNdxs.txt'),
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

def getNormalLoss(output_normal, normal, mask):
    '''
        compute the loss for the mini-batch
    '''
    mask = mask[:,0,:,:]
    # inner product 
    inner_product = torch.acos(torch.sum(output_normal*normal, dim=1))
    ind = torch.abs(mask - 1) < 1e-6
    total_error = torch.sum(torch.abs(inner_product*mask))
    total_pixel = torch.sum(torch.abs(mask))

    threshold_1 = 0.19634954084936207 # 11.25
    threshold_2 = 0.3883357585687383 # 22.5
    threshold_3 = 0.5235987755982988 # 30

    total_tr1 = torch.sum((inner_product < threshold_1).float()*mask)
    total_tr2 = torch.sum((inner_product < threshold_2).float()*mask)
    total_tr3 = torch.sum((inner_product < threshold_3).float()*mask)
    return total_error, total_tr1, total_tr2, total_tr3, total_pixel, inner_product[ind].view(-1)
    

def main(savePath, coarseModelPath, modelPath):
    begin_time = time.time()

    my_network = torch.load(os.path.join(modelPath, 'trained_model.t7'))
    my_network.cuda()
    my_network.train(False)

    global my_network_coarse 
    my_network_coarse = torch.load(os.path.join(coarseModelPath, 'trained_model.t7'))
    my_network_coarse.cuda()
    my_network_coarse.train(False)

    total_count = 0
    total_error = 0
    total_count_1 = 0
    total_count_2 = 0
    total_count_3 = 0
    all_error = None
    numIte = 0
    sphereNormal, _, _ = getSHNormal(64)
    sphereNormal = Variable(torch.Tensor(sphereNormal).cuda(), volatile=True).float()

    print("============================= test on NYU ============================")

    for ii, data in enumerate(testLoader, 0):
        inputs, normal, mask = data
        inputs, normal, mask = Variable(inputs.cuda(), volatile=True).float(),\
            Variable(normal.cuda(), volatile=True).float(), \
            Variable(mask.cuda(), volatile=True).float()

        # --------------------------------------------------------------------------------
        # get albedo, shading, normal, lighting in coarse scale and prepare the residual
        coarse_albedo, coarse_normal, coarse_lighting = \
                my_network_coarse(F.upsample(inputs, size=[64, 64], mode='bilinear'))
        output_normal = F.normalize((coarse_normal), p=2, dim=1)
        #coarse_shading = shadingLayer_coarse(coarse_normal, coarse_lighting)
        #coarse_albedo = F.upsample(coarse_albedo, size=[IMAGE_SIZE, IMAGE_SIZE], mode='bilinear')
        #coarse_normal = F.upsample(coarse_normal, size=[IMAGE_SIZE, IMAGE_SIZE], mode='bilinear')
        #coarse_shading = F.upsample(coarse_shading, size=[IMAGE_SIZE, IMAGE_SIZE], mode='bilinear')

        ## NOTE: we have a bug in coarse network for lighting, correct it
        #coarse_lighting = Variable(coarse_lighting[:,0:27].data).float()
        #coarse_lighting = coarse_lighting.unsqueeze(-1)
        #coarse_lighting = coarse_lighting.unsqueeze(-1)

        ## concatenate images, albedo, normal, shading as input
        #inputs_albedo = torch.cat((inputs, coarse_albedo), dim=1)
        #inputs_normal = torch.cat((inputs, coarse_normal), dim=1)
        #inputs_lighting = torch.cat((inputs, coarse_albedo, coarse_normal, coarse_shading), dim=1)

        #output_albedo, output_normal, output_lighting = my_network(inputs_albedo, inputs_normal, inputs_lighting)

        #true_lighting = output_lighting + coarse_lighting.expand(-1,-1, IMAGE_SIZE, IMAGE_SIZE)
        ## get shading
        #output_shading = shadingLayer(F.normalize(coarse_normal + output_normal, p=2, dim=1), 
        #        true_lighting)
        #
        #output_albedo = output_albedo + coarse_albedo
        #output_normal = F.normalize((output_normal + coarse_normal), p=2, dim=1)

        ## visualize the light
        #size_light = 64
        #numSamples = inputs.size()[0]
        ## compute the average of every 16x16 pixels for lighting
        #subArea = 8 
        #numH = IMAGE_SIZE/subArea
        #numW = IMAGE_SIZE/subArea
        #whole_sphere = Variable(torch.zeros(numSamples, 3, numH*size_light, numW*size_light).cuda()).float()
        #out_sphereNormal = sphereNormal.expand(numSamples, 3, size_light, size_light)
        #for i in range(numH):
        #    for j in range(numW):
        #        tmpLight = true_lighting[:,:,i*subArea:(i+1)*subArea, j*subArea: (j+1)*subArea]
        #        tmpLight = torch.mean(torch.mean(tmpLight, dim=-1), dim=-1)
        #        output_sphere = shadingLayer_coarse(out_sphereNormal, tmpLight)
        #        whole_sphere[:,:,i*size_light:(i+1)*size_light, j*size_light : (j+1)*size_light] = output_sphere
        #
        ## let's only record 25 batches, 500 images
        #if ii < 10:
        #    getImage(inputs, output_albedo, output_shading, whole_sphere, 
        #            output_normal, numIte, savePath)
        #numIte += 1
        error, count_1, count_2, count_3, count, error_list = getNormalLoss(output_normal, normal, mask)
        if all_error is None:
            all_error = error_list
        else:
            all_error = torch.cat((all_error, error_list), dim=0)
        total_error += error
        total_count_1 += count_1
        total_count_2 += count_2
        total_count_3 += count_3
        total_count += count
    total_error = total_error.data.cpu().numpy()/total_count.data.cpu().numpy()
    medianAngle = torch.median(all_error).data[0]
    total_count_1 = total_count_1.data.cpu().numpy()/total_count.data.cpu().numpy()
    total_count_2 = total_count_2.data.cpu().numpy()/total_count.data.cpu().numpy()
    total_count_3 = total_count_3.data.cpu().numpy()/total_count.data.cpu().numpy()
    print total_count_1
    print total_count_2
    print total_count_3
    print 'average angle is {:0.4f}'.format(total_error[0]/np.pi*180.0)
    print 'median angle is {:0.4f}'.format(medianAngle/np.pi*180.0)
    print 'number of pixel smaller than 11.25 is {:0.8f}'.format(total_count_1[0])
    print 'number of pixel smaller than 22.5 is {:0.8f}'.format(total_count_2[0])
    print 'number of pixel smaller than 30 is {:0.8f}'.format(total_count_3[0])


if __name__ == '__main__':
    savePath = sys.argv[1]
    coarseModelPath = sys.argv[2]
    modelPath = sys.argv[3]
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    main(savePath, coarseModelPath, modelPath)
