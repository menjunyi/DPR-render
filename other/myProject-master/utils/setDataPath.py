''' 
    define data loader
'''
import os
import sys
import copy
sys.path.append('utils')
from loadData_real import *

SAW_path = 'SAW/pixel_label/saw_data-filter_size_0-ignore_border_0.05-normal_gradmag_thres_1.5-depth_gradmag_thres_2.0'
# for training
# set data path for suncg data
baseSUNCG = '/scratch1/intrinsicImage/synthetic_SUNCG/'
SUNCG_data = {}
#SUNCG_data['fileNameList'] = os.path.join(baseSUNCG, 'training.list')
SUNCG_data['fileNameList'] = None
SUNCG_data['dataFolder'] = os.path.join(baseSUNCG, 'images_color')
SUNCG_data['albedoFolder'] = os.path.join(baseSUNCG, 'albedo')
SUNCG_data['shadingFolder'] = os.path.join(baseSUNCG, 'shading')
SUNCG_data['normalFolder'] = '/scratch1/data/SUNCG_groundTruth/SUNCG_normal'
SUNCG_data['maskFolder'] =  os.path.join(baseSUNCG, 'images_mask_correct')
SUNCG_data['labelFolder'] = os.path.join(baseSUNCG, 'pixel_order_albedo_mean')

# for IIW
baseReal = '/scratch1/data/IIW/realData_cvpr2019/'
IIW_data = {}
IIW_data['fileNameList'] = None
IIW_data['dataFolder'] = os.path.join(baseReal, 'IIW', 'data')
IIW_data['SAW'] = os.path.join(baseReal, SAW_path) 
#IIW_data['labelFolder'] = os.path.join(baseReal, 'IIW', 'label_augmented_forPytorch')
IIW_data['labelFolder'] = os.path.join(baseReal, 'IIW', 'label_forPytorch')

# for NYU dataset
NYU_data = {}
NYU_data['fileNameList'] = None
NYU_data['dataFolder'] = os.path.join(baseReal, 'NYU', 'image')
NYU_data['normalFolder'] = os.path.join(baseReal, 'NYU', 'normal')
NYU_data['normalMask'] = os.path.join(baseReal, 'NYU', 'normal')
NYU_data['SAW'] = os.path.join(baseReal, SAW_path)

def constructDataLoader(dataType, DA_type, imgSize, expand = 10, ind_rotate=True):
    '''
        construct data loader
    '''
    transform = dataAugmentation(imgSize, expand, ind_rotate)
    SUNCG_data_instance = copy.deepcopy(SUNCG_data)
    IIW_data_instance = copy.deepcopy(IIW_data)
    NYU_data_instance = copy.deepcopy(NYU_data)
    if dataType == 'train':
        SUNCG_data_instance['fileNameList'] = os.path.join(baseSUNCG, 'training.list')
        IIW_data_instance['fileNameList'] = os.path.join(baseReal, 'IIW', 'IIW_train.list')
        NYU_data_instance['fileNameList'] = os.path.join(baseReal, 'NYU', 'correspondence_train.list')
    elif dataType == 'val':
        SUNCG_data_instance['fileNameList'] = os.path.join(baseSUNCG, 'testing.list')
        IIW_data_instance['fileNameList'] = os.path.join(baseReal, 'IIW', 'IIW_val.list')
        NYU_data_instance['fileNameList'] = os.path.join(baseReal, 'NYU', 'correspondence_val.list')
    elif dataType == 'test':
        SUNCG_data_instance['fileNameList'] = os.path.join(baseSUNCG, 'testing.list')
        IIW_data_instance['fileNameList'] = os.path.join(baseReal, 'IIW', 'IIW_test.list')
        NYU_data_instance['fileNameList'] = os.path.join(baseReal, 'NYU', 'correspondence_test.list')

    return loadReal(SUNCG_data_instance, IIW_data_instance, NYU_data_instance, DA_type, transform)
