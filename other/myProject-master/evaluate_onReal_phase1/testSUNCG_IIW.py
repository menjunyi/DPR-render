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
from defineDRN_normal import *
from defineHourglass_64 import *
from utils_shading import *
from defineLoss_IIW_CGI import *
from loadIIW import *
from defineHelp import *
from defineHelp_lighting import *

# set random seed for all possible randomness
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

my_network = DRN(6, 6, 12)
my_network.cuda()
my_network_coarse = HourglassNet(27)
my_network_coarse.cuda()


full_root = '/scratch1/data/IIW/iiw-dataset/data/'
test_list_dir = '../evaluation/'

IMAGE_SIZE=128

gradientLayer = gradientLayer_color()
gradientLayer.cuda()
shadingLayer_coarse = constructShading()
shadingLayer_coarse.cuda()
shadingLayer = constructShading_lightImg()
shadingLayer.cuda()
samplingLightLayer = samplingLight()
samplingLightLayer.cuda()

#testLoaderHelper = IIWTESTDataLoader(full_root, test_list_hdr, IMAGE_SIZE, )
#
#testLoader = torch.utils.data.DataLoader(testLoaderHelper, 
#    batch_size=20, shuffle=False, num_workers=5)

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

def main(savePath, coarseModelPath, modelPath):
    begin_time = time.time()

    #my_network = torch.load(os.path.join(modelPath, 'trained_model.t7'))
    #my_network.cuda()
    #my_network.train(False)

    #global my_network_coarse 
    #my_network_coarse = torch.load(os.path.join(coarseModelPath, 'trained_model.t7'))
    #my_network_coarse.cuda()
    #my_network_coarse.train(False)

    my_network.load_state_dict(torch.load(os.path.join(modelPath, 'trained_model', 'trained_model_08.t7')))
    my_network.cuda()
    my_network.train(False)

    my_network_coarse.load_state_dict(torch.load(os.path.join(coarseModelPath, 'trained_model', 'trained_model_19.t7')))
    my_network_coarse.cuda()
    my_network_coarse.train(False)

    test_loss = 0
    total_count = 0
    numIte = 0
    sphereNormal, _, _ = getSHNormal(64)
    #fid = open('../SAW_trainng_CGI.list', 'w')
    sphereNormal = Variable(torch.Tensor(sphereNormal).cuda(), volatile=True).float()
    for mode in range(0,3):
        print("============================= Validation IIW TESTSET ============================", mode)
        data_loader_IIW_TEST = IIWTESTDataLoader(full_root, test_list_dir, IMAGE_SIZE, mode)
        dataset_iiw_test = data_loader_IIW_TEST.load_data()

        for ii, data in enumerate(dataset_iiw_test, 0):
            inputs, targets, image_id = data
            #eor item in image_id:
            #    print>>fid, '{:d}'.format(item)
            inputs = Variable(inputs.cuda(), volatile=True).float()

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
            inputs_albedo = torch.cat((inputs, coarse_albedo), dim=1)
            inputs_normal = torch.cat((inputs, coarse_normal), dim=1)
            inputs_lighting = torch.cat((inputs, coarse_albedo, coarse_normal, coarse_shading), dim=1)

            output_albedo, output_normal, output_lighting = my_network(inputs_albedo, inputs_normal, inputs_lighting)

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
            if ii < 10:
                getImage(inputs, output_albedo, output_shading, whole_sphere, 
                        output_normal, numIte, savePath)
            numIte += 1
            total_whdr, count = evaluate_WHDR(output_albedo, targets)
            test_loss += total_whdr
            total_count += count
        print 'total_loss is {:0.4f} number of image is {:0.4f}'.format(test_loss, total_count)
        print 'average whdr is {:0.4f}'.format(test_loss/total_count)
    #fid.close()


if __name__ == '__main__':
    savePath = sys.argv[1]
    coarseModelPath = sys.argv[2]
    modelPath = sys.argv[3]
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    main(savePath, coarseModelPath, modelPath)
