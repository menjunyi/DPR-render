import sys
sys.path.append('../utils')
sys.path.append('../model')
sys.path.append('../evaluation')

import os
import numpy as np
import time

from torch.autograd import Variable
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F

# load module defined by myself
#from defineHourglass_128 import *
from utils_shading import *
from defineLoss_IIW import *
from loadIIW import *
from defineHelp import *
from defineHelp_lighting import *
from help_saw import *
from loadSAW_256 import *

# set random seed for all possible randomness
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
global my_network_coarse
global my_network_fine
global my_network_detail
global shadingLayer
global shadingLayer_coarse
shadingLayer_coarse = constructShading()
shadingLayer_coarse.cuda()
shadingLayer = constructShading_lightImg()
shadingLayer.cuda()


full_root = '/scratch1/data/IIW/saw/'
test_list_dir = '../evaluation/'

IMAGE_SIZE=256

def main(savePath, coarseModelPath, fineModelPath, detailModelPath):
    begin_time = time.time()

    my_network_detail = torch.load(os.path.join(detailModelPath, 'trained_model.t7'))
    my_network_detail.cuda()
    my_network_detail.train(False)

    my_network_fine = torch.load(os.path.join(fineModelPath, 'trained_model.t7'))
    my_network_fine.cuda()
    my_network_fine.train(False)

    my_network_coarse = torch.load(os.path.join(coarseModelPath, 'trained_model.t7'))
    my_network_coarse.cuda()
    my_network_coarse.train(False)
    #-------------------------------------------------------------------------------------------------------
    pixel_labels_dir = full_root + 'saw_pixel_labels/saw_data-filter_size_0-ignore_border_0.05-normal_gradmag_thres_1.5-depth_gradmag_thres_2.0'
    splits_dir = full_root + 'saw_splits/'
    img_dir = full_root + "saw_images_512/"
    dataset_split = 'E'
    class_weights = [1, 1, 2]
    bl_filter_size = 10
    print("============================= Validation ON SAW============================")
    AP = compute_pr(pixel_labels_dir, splits_dir,
                dataset_split, class_weights, bl_filter_size, img_dir,
                [my_network_coarse, my_network_fine, my_network_detail, shadingLayer_coarse, shadingLayer])
    print("Current AP: %f"%AP)
    return AP

if __name__ == '__main__':
    savePath = sys.argv[1]
    coarseModelPath = sys.argv[2]
    fineModelPath = sys.argv[3]
    detailModelPath = sys.argv[4]
    
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    main(savePath, coarseModelPath, fineModelPath, detailModelPath)
