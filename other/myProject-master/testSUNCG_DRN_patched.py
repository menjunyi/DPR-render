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
from defineDRN_patched import *
import utils_shading
from defineCriterion import *
from loadData_basic import *
from defineHelp import *
import helpDRN_patched_overlap
from defineHelp_lighting import *

# set random seed for all possible randomness
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


baseFolder = '/scratch1/intrinsicImage/synthetic_SUNCG/'

#criterion = MSELoss()
criterion = mask_loss()

# image size to deal with
IMAGE_SIZE=120
PATCH_SIZE = 16
STRIDE_SIZE= 8

my_network = DRN(6,6, 12)
my_network.cuda()
gradientLayer = gradientLayer_color()
gradientLayer.cuda()
shadingLayer_coarse = utils_shading.constructShading()
shadingLayer_coarse.cuda()
shadingLayer_fine = helpDRN_patched_overlap.constructShading(
        IMAGE_SIZE, PATCH_SIZE, STRIDE_SIZE)
shadingLayer_fine.cuda()
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

def getImage(image, output_albedo, output_shading, output_sphere, output_normal, diff_normal, count, resultPath):
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
				diffNormal = np.squeeze(diff_normal[i].cpu().data.numpy())

				np.save(os.path.join(resultPath, 'normal_{:04d}_{:02d}.npy'.format(count, i)), imgNormal)

				imgAlbedo = imgAlbedo.transpose((1,2,0))
				imgShading = imgShading.transpose((1,2,0))
				imgSphere = imgSphere.transpose((1,2,0))
				imgNormal = imgNormal.transpose((1,2,0))
				imgNormal = imgNormal / (np.linalg.norm(imgNormal, axis=-1, keepdims=True) + 1e-6)
				diffNormal = diffNormal.transpose((1,2,0))

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
				diffNormal = ((np.abs(diffNormal)/2.0)*255).astype(np.uint8)
				print diffNormal.shape

				io.imsave(os.path.join(resultPath, 'shading_{:04d}_{:02d}.png'.format(count, i)), imgShading)
				io.imsave(os.path.join(resultPath, 'albedo_{:04d}_{:02d}.png'.format(count, i)), imgAlbedo)
				io.imsave(os.path.join(resultPath, 'sphere_{:04d}_{:02d}.png'.format(count, i)), imgSphere)
				io.imsave(os.path.join(resultPath, 'normal_{:04d}_{:02d}.png'.format(count, i)), imgNormal)
				io.imsave(os.path.join(resultPath, 'diffNormal_{:04d}_{:02d}.png'.format(count, i)), diffNormal)

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

		#loss_normal = criterion(mask, torch.sum(-1*output_normal*normal, dim=1, keepdim=True))
		loss_normal = criterion(imgMask, torch.abs(output_normal - normal))
		loss_normal_grad = criterion(gradMask, torch.abs(normal_grad - output_normal_grad))

		#return loss_albedo + loss_shading + loss_albedo_grad + loss_shading_grad + loss_normal + loss_normal_grad
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
    sphereNormal = np.squeeze(sphereNormal)
    sphereNormal = Variable(torch.Tensor(sphereNormal).unsqueeze(0).cuda(), volatile=True).float()
    
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
        inputs_albedo = torch.cat((inputs, coarse_albedo), dim=1)
        inputs_normal = torch.cat((inputs, coarse_normal), dim=1)
        inputs_lighting = torch.cat((inputs, coarse_albedo, coarse_normal, coarse_shading), dim=1)

        # predict residual
        output_albedo, output_normal, light_UL, \
                light_UR, light_DL, light_DR \
                = my_network(inputs_albedo, inputs_normal, inputs_lighting)

        true_albedo = output_albedo + coarse_albedo
        true_normal = F.normalize(output_normal + coarse_normal, p=2, dim=1)
        true_shading = shadingLayer_fine(light_UL, light_UR, light_DL, light_DR, true_normal)


        # combine lighting
        numSamples = light_UL.shape[0]
        numH = light_UL.shape[2]
        numW = light_UL.shape[3]
        output_lighting = Variable(torch.zeros(numSamples, 27, numH*2, numW*2).cuda()).float()
        for i in range(numH):
            for j in range(numW):
                output_lighting[:,:,2*i,2*j] = light_UL[:,:,i,j]
                output_lighting[:,:,2*i+1,2*j] = light_DL[:,:,i,j]
                output_lighting[:,:,2*i,2*j+1] = light_UR[:,:,i,j]
                output_lighting[:,:,2*i+1,2*j+1] = light_DR[:,:,i,j]
        print 'output_ligint'
        print output_lighting.shape

        size_light = 64
        numSamples = output_lighting.shape[0]
        numH = output_lighting.shape[2]
        numW = output_lighting.shape[3]
        whole_sphere = Variable(torch.zeros(numSamples, 3, numH*size_light, numW*size_light).cuda()).float()
        out_sphereNormal = sphereNormal.expand(numSamples, 3, size_light, size_light)
        for i in range(numH):
            for j in range(numW):
                print output_lighting[:,:,i,j].shape
                tmp_sphere = shadingLayer_coarse(out_sphereNormal, output_lighting[:,:,i,j])
                whole_sphere[:, :, i*size_light:(i+1)*size_light, j*size_light:(j+1)*size_light] = tmp_sphere

        # let's only record 25 batches, 500 images
        if ii < 25:
            getImage(inputs, true_albedo, true_shading, whole_sphere, true_normal, output_normal, count, savePath)

        loss_albedo, loss_shading, loss_albedo_grad, loss_shading_grad, loss_normal, loss_normal_grad = \
               getLoss(inputs, true_albedo, true_shading, true_normal, albedo, shading, normal, mask)
        
        tl_albedo += loss_albedo
        tl_shading += loss_shading
        tl_albedo_grad += loss_albedo_grad
        tl_shading_grad += loss_shading_grad
        tl_normal_grad += loss_normal_grad
        
        count += 1
    print 'albedo loss is %.4f' % (tl_albedo.data.cpu().numpy()/count)
    print 'albedo gradient loss is %.4f' % (tl_albedo_grad.data.cpu().numpy()/count)
    print 'shading loss is %.4f' % (tl_shading.data.cpu().numpy()/count)
    print 'shading gradient loss is %.4f' % (tl_shading_grad.data.cpu().numpy()/count)
    print 'normal loss is %.4f' % (tl_normal.data.cpu().numpy()/count)
    print 'normal gradient loss is %.4f' % (tl_normal_grad.data.cpu().numpy()/count)

if __name__ == '__main__':
    savePath = sys.argv[1]
    coarseModelPath = sys.argv[2]
    modelPath = sys.argv[3]
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    main(savePath, coarseModelPath, modelPath)
