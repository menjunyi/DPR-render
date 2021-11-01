import sys
sys.path.append('../utils')
sys.path.append('../model')
from skimage import io

import os
import numpy as np
import time

import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch
from tensorboardX import SummaryWriter

from setDataPath import *

# set random seed for all possible randomness
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

trainLoaderHelper_compare = constructDataLoader('train', 'compare', 256)
trainLoaderHelper_shading = constructDataLoader('train', 'shading', 256)

trainLoader_compare = torch.utils.data.DataLoader(trainLoaderHelper_compare, 
	batch_size=2, shuffle=False, num_workers=5)

trainLoader_shading = torch.utils.data.DataLoader(trainLoaderHelper_shading,
	batch_size=2, shuffle=False, num_workers=5)
shading_iter = iter(trainLoader_shading)

for (i, data_compare) in enumerate(trainLoader_compare, 0):
    if i > 0:
        break
    data_shading = next(shading_iter, None)
    #print data_compare
    #print data_shading

    # save loaded images for debug
    path_1 = 'compare'
    if not os.path.exists(path_1):
        os.makedirs(path_1)
    numImages = data_compare[0].shape[0]
    for i in range(numImages):
        io.imsave(os.path.join(path_1, 'suncg_{:02d}.png'.format(i)), (255*data_compare[0][i].numpy()).astype(np.uint8))
        io.imsave(os.path.join(path_1, 'iiw_{:02d}.png'.format(i)), (255*data_compare[1][i].numpy()).astype(np.uint8))
        io.imsave(os.path.join(path_1, 'nyu_{:02d}.png'.format(i)), (255*data_compare[2][i].numpy()).astype(np.uint8))
        io.imsave(os.path.join(path_1, 'suncg_albedo_{:02d}.png'.format(i)), (255*data_compare[3]['albedo'][i].numpy()).astype(np.uint8))
        io.imsave(os.path.join(path_1, 'suncg_shading_{:02d}.png'.format(i)), (255*data_compare[3]['shading'][i].numpy()).astype(np.uint8))
        io.imsave(os.path.join(path_1, 'suncg_normal_{:02d}.png'.format(i)), (255*(data_compare[3]['normal'][i].numpy()+1)/2).astype(np.uint8))
        io.imsave(os.path.join(path_1, 'suncg_mask_{:02d}.png'.format(i)), (255*data_compare[3]['mask'][i].numpy()).astype(np.uint8))
        print '.....................'
        print data_compare[3]['labelName'][i]
        io.imsave(os.path.join(path_1, 'iiw_mask_1_{:02d}.png'.format(i)), (255.0/data_compare[4]['num_saw_mask_1'][i].numpy()*data_compare[4]['saw_mask_1'][i].numpy()).astype(np.uint8))
        io.imsave(os.path.join(path_1, 'iiw_mask_2_{:02d}.png'.format(i)), (255.0/data_compare[4]['num_saw_mask_2'][i].numpy()*data_compare[4]['saw_mask_2'][i].numpy()).astype(np.uint8))
        print data_compare[4]['labelName'][i]
        
        io.imsave(os.path.join(path_1, 'nyu_normal_{:02d}.png'.format(i)), (255*(data_compare[5]['normal'][i].numpy()+1)/2).astype(np.uint8))
        io.imsave(os.path.join(path_1, 'nyu_mask_{:02d}.png'.format(i)), (255*data_compare[5]['mask'][i].numpy()).astype(np.uint8))
        io.imsave(os.path.join(path_1, 'nyu_mask_1_{:02d}.png'.format(i)), (255.0/data_compare[5]['num_saw_mask_1'][i].numpy()*data_compare[5]['saw_mask_1'][i].numpy()).astype(np.uint8))
        io.imsave(os.path.join(path_1, 'nyu_mask_2_{:02d}.png'.format(i)), (255.0/data_compare[5]['num_saw_mask_2'][i].numpy()*data_compare[5]['saw_mask_2'][i].numpy()).astype(np.uint8))
    # save loaded images for debug
    path_2 = 'saw'
    if not os.path.exists(path_2):
        os.makedirs(path_2)
    numImages = data_shading[0].shape[0]
    for i in range(numImages):
        io.imsave(os.path.join(path_2, 'suncg_{:02d}.png'.format(i)), (255*data_shading[0][i].numpy()).astype(np.uint8))
        io.imsave(os.path.join(path_2, 'iiw_{:02d}.png'.format(i)), (255*data_shading[1][i].numpy()).astype(np.uint8))
        io.imsave(os.path.join(path_2, 'nyu_{:02d}.png'.format(i)), (255*data_shading[2][i].numpy()).astype(np.uint8))
        io.imsave(os.path.join(path_2, 'suncg_albedo_{:02d}.png'.format(i)), (255*data_shading[3]['albedo'][i].numpy()).astype(np.uint8))
        io.imsave(os.path.join(path_2, 'suncg_shading_{:02d}.png'.format(i)), (255*data_shading[3]['shading'][i].numpy()).astype(np.uint8))
        io.imsave(os.path.join(path_2, 'suncg_normal_{:02d}.png'.format(i)), (255*(data_shading[3]['normal'][i].numpy()+1)/2).astype(np.uint8))
        io.imsave(os.path.join(path_2, 'suncg_mask_{:02d}.png'.format(i)), (255*data_shading[3]['mask'][i].numpy()).astype(np.uint8))
        print '.....................'
        print data_shading[3]['labelName'][i]
        io.imsave(os.path.join(path_2, 'iiw_mask_1_{:02d}.png'.format(i)), (255.0/data_shading[4]['num_saw_mask_1'][i].numpy()*data_shading[4]['saw_mask_1'][i].numpy()).astype(np.uint8))
        io.imsave(os.path.join(path_2, 'iiw_mask_2_{:02d}.png'.format(i)), (255.0/data_shading[4]['num_saw_mask_2'][i].numpy()*data_shading[4]['saw_mask_2'][i].numpy()).astype(np.uint8))
        print data_shading[4]['labelName'][i]
        
        io.imsave(os.path.join(path_2, 'nyu_normal_{:02d}.png'.format(i)), (255*(data_shading[5]['normal'][i].numpy()+1)/2).astype(np.uint8))
        io.imsave(os.path.join(path_2, 'nyu_mask_{:02d}.png'.format(i)), (255*data_shading[5]['mask'][i].numpy()).astype(np.uint8))
        io.imsave(os.path.join(path_2, 'nyu_mask_1_{:02d}.png'.format(i)), (255.0/data_shading[5]['num_saw_mask_1'][i].numpy()*data_shading[5]['saw_mask_1'][i].numpy()).astype(np.uint8))
        io.imsave(os.path.join(path_2, 'nyu_mask_2_{:02d}.png'.format(i)), (255.0/data_shading[5]['num_saw_mask_2'][i].numpy()*data_shading[5]['saw_mask_2'][i].numpy()).astype(np.uint8))

