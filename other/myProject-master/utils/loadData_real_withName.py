import os
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import skimage.morphology
from skimage.morphology import square
import numpy as np
import random
import pandas as pd
random.seed(0)
from scipy.ndimage.measurements import label as fun_label
import cv2

'''
    load data for fine-tuning on real data
'''

class loadReal(Dataset):
    '''
        load real data
    '''
    def __init__(self, SUNCG_data, IIW_data, NYU_data, DA_type, transform=None):
        '''
            data contains three dataset
            SUNCG synthetic data, contains ground truth albedo, shading, and normal
            IIW_data: intrinsic image in the wild, contains ground truth WHDR, shaing information
            NYU_data: NYU dataset, contains groundtruth normal, shading information
        '''
        self.SUNCG_list = []
        self.IIW_list = []
        self.NYU_list = []
        with open(SUNCG_data['fileNameList']) as f:
            for line in f:
                self.SUNCG_list.append(line.strip())
        with open(IIW_data['fileNameList']) as f:
            for line in f:
                self.IIW_list.append(line.strip())
        with open(NYU_data['fileNameList']) as f:
            for line in f:
                # NOTE: we have two file names for each NYU data,
                # One in original NYU dataset, used to load image and normal
                # The other in SAW dataset, used to load shading
                self.NYU_list.append(line.strip().split())
        self.SUNCG_dic = SUNCG_data
        self.IIW_dic = IIW_data
        self.NYU_dic = NYU_data
        self.transform = transform
        self.DA_type = DA_type
        self.lenNYU = len(self.NYU_list)
        self.numCompare = 100

    def __len__(self):
        # since the three datasets have different number of samples
        # we use number of IIW images as the length of the dataset
        return len(self.IIW_list)

    def __getitem__(self, idx):
        '''
            load one sample for each of the data set
        '''
        # load data for synthetic images
        label_SUNCG = {}
        image_SUNCG = np.array([])
        if self.SUNCG_dic:
            image_SUNCG, label_SUNCG['albedo'], label_SUNCG['shading'], \
                    label_SUNCG['normal'], label_SUNCG['mask'], label_SUNCG['label'] \
                    = self.load_synthetic(idx)
            if self.transform:
                image_SUNCG, label_SUNCG = self.transform(image_SUNCG, label_SUNCG, self.DA_type)

        # load data for IIW
        label_IIW = {}
        image_IIW = np.array([])
        if self.IIW_dic:
            label_IIW = {}
            image_IIW, label_IIW['saw_mask_1'], label_IIW['saw_mask_2'], \
                    label_IIW['label'] = self.load_IIW(idx)
            if self.transform:
                image_IIW, label_IIW = self.transform(image_IIW, label_IIW, self.DA_type)

        # load data for NYU
        label_NYU = {}
        image_NYU = np.array([])
        if self.NYU_dic:
            label_NYU = {}
            image_NYU, label_NYU['normal'], label_NYU['mask'],\
                    label_NYU['saw_mask_1'], label_NYU['saw_mask_2'] = self.load_NYU(idx)
            if self.transform:
                image_NYU, label_NYU = self.transform(image_NYU, label_NYU, self.DA_type)

        return image_SUNCG, image_IIW, image_NYU, label_SUNCG, label_IIW, label_NYU, self.IIW_list[idx]
    
    def loadLabel(self, labelName, dataName):
        '''
            load csv file 
        '''
        data_csv = pd.read_csv(labelName).values
        data_csv = data_csv[:,2:]

        if dataName == 'SUNCG':
            # for suncg dataset, select 3000 comparisons
            tmpLabel = data_csv[:,-2]
            ind_equal = np.where(tmpLabel == 0)[0]
            ind_darker = np.where(tmpLabel == 1)[0]
            ind_lighter = np.where(tmpLabel == 2)[0]
            if ind_equal.shape[0] > self.numCompare:
                np.random.shuffle(ind_equal)
                ind_equal = ind_equal[0:self.numCompare]
            if ind_darker.shape[0] > self.numCompare:
                np.random.shuffle(ind_darker)
                ind_darker = ind_darker[0:self.numCompare]
            if ind_lighter.shape[0] > self.numCompare:
                np.random.shuffle(ind_lighter)
                ind_lighter = ind_lighter[0:self.numCompare]
            data_csv = np.concatenate((data_csv[ind_equal], data_csv[ind_darker],
                data_csv[ind_lighter]), axis=0)

        return data_csv

    def load_synthetic(self, idx):
        '''
            load one data for SUNCG
            load data for our synthesized data
            including: image, albedo, shading, normal, mask, comparison
        '''
        fileName = self.SUNCG_list[idx]
        # load image, albedo, shading, normal and mask
        imgName = os.path.join(self.SUNCG_dic['dataFolder'], fileName)
        image = io.imread(imgName)
        if len(image.shape)==2:
            image = np.tile(image[...,None], (1, 3))
        albedoName = os.path.join(self.SUNCG_dic['albedoFolder'], fileName)
        albedo = io.imread(albedoName)
        if len(albedo.shape)==2:
            albedo = np.tile(albedo[...,None], (1, 3))
        shadingName = os.path.join(self.SUNCG_dic['shadingFolder'], fileName)
        shading = io.imread(shadingName)
        if len(shading.shape)==2:
            shading = np.tile(shading[...,None], (1, 3))
        normalName = os.path.join(self.SUNCG_dic['normalFolder'], fileName[0:-8] + '_norm_camera.png')
        normal = io.imread(normalName)
        
        normalMaskName = os.path.join(self.SUNCG_dic['normalFolder'], fileName[0:-8] + '_valid.png')
        normalMask = io.imread(normalMaskName)
        normalMask = np.expand_dims(normalMask, axis=-1)
        
        maskName = os.path.join(self.SUNCG_dic['maskFolder'], fileName)
        mask = io.imread(maskName)
        mask = cv2.resize(mask, tuple(normalMask.shape[1::-1]), interpolation=cv2.INTER_CUBIC)
        mask = np.expand_dims(mask, axis=-1)

        labelName = os.path.join(self.SUNCG_dic['labelFolder'], fileName[0:-4] + '.csv')

        # pre-process data
        image = 1.0*image/255.0
        albedo = 1.0*albedo/255.0
        shading = 1.0*shading/255.0
        normal = normal.astype(np.float)
        normal = (normal/255.0-0.5)*2
        normal = normal/(np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6)
        mask = 1.0*mask/255.0
        normalMask = 1.0*normalMask/255.0
        mask = mask*normalMask
        return image, albedo, shading, normal, mask, labelName
        #return image, albedo, shading, normal, mask, label

    def load_IIW(self, idx):
        '''
            load images and labels for IIW  (and labels for SAW)
        '''
        fileName = self.IIW_list[idx]
        imgName = os.path.join(self.IIW_dic['dataFolder'], fileName + '.png')
        image = io.imread(imgName)
        if len(image.shape)==2:
            image = np.tile(image[...,None], (1, 3))

        # prepare shading mask
        shadingName = os.path.join(self.IIW_dic['SAW'], fileName + '.npy')
        sawShading = np.load(shadingName)
        saw_mask_1, saw_mask_2 = self.processShadingMask(sawShading)

        labelName = os.path.join(self.IIW_dic['labelFolder'], fileName + '.csv')
        #label = self.loadLabel(labelName, 'IIW')

        image = 1.0*image/255.0
        return image, saw_mask_1, saw_mask_2, labelName
        #return image, saw_mask_1, saw_mask_2, label

    def load_NYU(self, idx):
        '''
            load images, normals and mask for NYU dataset
        '''
        idx = idx % self.lenNYU
        fileName_SAW = self.NYU_list[idx][0]
        fileName_NYU = self.NYU_list[idx][1]
        imgName = os.path.join(self.NYU_dic['dataFolder'], fileName_NYU + '.png')
        image = io.imread(imgName)
        if len(image.shape)==2:
            image = np.tile(image[...,None], (1,3))
        normalName = os.path.join(self.NYU_dic['normalFolder'], fileName_NYU + '_normal.png')
        normal = io.imread(normalName)
        maskName = os.path.join(self.NYU_dic['normalMask'], fileName_NYU + '_mask.png')
        mask = io.imread(maskName)
        mask = np.expand_dims(mask, axis=-1)
        shadingName = os.path.join(self.NYU_dic['SAW'], fileName_SAW + '.npy')
        shading = np.load(shadingName)
        saw_mask_1, saw_mask_2 = self.processShadingMask(shading)

        image = 1.0*image/255.0
        mask = 1.0*mask/255.0
        normal = normal.astype(np.float)
        normal = (normal/255.0-0.5)*2
        normal = normal/(np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6)

        return image, normal, mask, saw_mask_1, saw_mask_2

    def processShadingMask(self, sawShading):
        '''
            filtering sawShading
        '''
        # NOTE: saw_mask_1 is the boundary of the shadow
        #       saw_mask_2 is the smooth region of the shadow
        saw_mask_1 = (sawShading == 1)
        saw_mask_1 = skimage.morphology.binary_dilation(saw_mask_1, square(9)).astype(np.float32)

        saw_mask_2 = (sawShading == 2).astype(np.float32)
        return saw_mask_1, saw_mask_2

class dataAugmentation(object):
    '''
        define data augmentation for non-comparison purpose
    '''
    def __init__(self, output_size, expand=10, ind_rotate=True):
        self.size = output_size
        self.expand = expand
        self.ind_rotate = ind_rotate

    def processImage(self, image, sH, sW, H, W, rotate_M): 
        # rotate
        imgHeight, imgWidth = image.shape[0:2]
        image = cv2.warpAffine(image, rotate_M, (imgWidth, imgHeight))
        # resize and random crop
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_CUBIC)
        if len(image.shape) < 3:
            image = image[sH:sH+self.size, sW:sW+self.size]
        else:
            image = image[sH:sH+self.size, sW:sW+self.size, :]
        return image

    def __call__(self, image, label, DA_type):
        '''
            random crop, in-plane rotaion for images 
            and apply same data augmentation for the labels
        '''
        imgHeight, imgWidth,_ = image.shape
        if not DA_type == 'compare':
            # first rotate
            if self.ind_rotate:
                if random.random() > 0.5:
                    # random rotate -10 to 10 degrees
                    angle = -10*random.random()
                    rotate_M = cv2.getRotationMatrix2D((imgHeight/2.0, imgWidth/2.0), angle, 1)
                else:
                    rotate_M = cv2.getRotationMatrix2D((self.size/2, self.size/2), 0, 1)
            # random crop
            H = self.size + self.expand
            W = self.size + self.expand
            maxH = H - self.size
            maxW = W - self.size
            sH = random.randint(0, maxH)
            sW = random.randint(0, maxW)
        else:
            # rotate 0 degree
            rotate_M = cv2.getRotationMatrix2D((self.size/2, self.size/2), 0, 1)

            # no resize and crop
            H = self.size
            W = self.size
            sH = 0
            sW = 0

        # preprocess image
        image = self.processImage(image, sH, sW, H, W, rotate_M)
        image = np.transpose(image, (2,0,1))
        ind = image < 0
        image[ind] = 0
        if 'albedo' in label.keys():
            tmp = self.processImage(label['albedo'], sH, sW, H, W, rotate_M)
            ind = tmp < 0
            tmp[ind] = 0
            label['albedo'] = np.transpose(tmp, (2,0,1))
        if 'shading' in label.keys():
            tmp = self.processImage(label['shading'], sH, sW, H, W, rotate_M)
            ind = tmp < 0
            tmp[ind] = 0
            label['shading'] = np.transpose(tmp, (2,0,1))
        if 'saw_mask_1' in label.keys():
            tmp_mask = self.processImage(label['saw_mask_1'], sH, sW, H, W, rotate_M)
            tmp_mask, numLabel = fun_label(tmp_mask)
            label['saw_mask_1'] = np.expand_dims(tmp_mask, axis=0)
            label['num_saw_mask_1'] = np.array([numLabel])
        if 'saw_mask_2' in label.keys():
            tmp_mask = self.processImage(label['saw_mask_2'], sH, sW, H, W, rotate_M)
            tmp_mask, numLabel = fun_label(tmp_mask)
            label['saw_mask_2'] = np.expand_dims(tmp_mask, axis=0)
            label['num_saw_mask_2'] = np.array([numLabel])
        if 'mask' in label.keys():
            tmp_mask = self.processImage(label['mask'], sH, sW, H, W, rotate_M)
            tmp_mask = (tmp_mask==1.0).astype(np.float)
            label['mask'] = np.expand_dims(tmp_mask, axis=0)
        if 'normal' in label.keys():
            # in-plane rotation
            #rotate_M = cv2.getRotationMatrix2D((self.size/2, self.size/2), 0, 1)
            tmpNormal = self.processImage(label['normal'], sH, sW, H, W, rotate_M)
            maskInd = np.sum(np.abs(tmpNormal), axis=2) == 0
            maskInd = np.tile(np.expand_dims(maskInd, -1), (1,1,3))
            # we need to rotate normal 
            # NOTE: 0 and 2 coordinate represent normal along x and y plane
            #if DA_type=='compare':
            #    rotate_M = cv2.getRotationMatrix2D((self.size/2, self.size/2), 0, 1)
            #else:
            #    rotate_M = cv2.getRotationMatrix2D((self.size/2, self.size/2), 90, 1)
            tmpNormal_xy = np.reshape(tmpNormal[:,:,0:3:2], (-1,2))
            tmpNormal_xy = np.dot(tmpNormal_xy, np.transpose(rotate_M[0:2, 0:2], (1,0)))
            tmpNormal_xy = np.reshape(tmpNormal_xy, (self.size, self.size, 2))
            tmpNormal[:,:,0:3:2] = tmpNormal_xy
            tmpNormal[maskInd] = 0
            tmpNormal = tmpNormal / (np.tile(np.sqrt(np.sum(tmpNormal**2, axis=2, keepdims=True)), (1,1,3)) + 1e-6)
            label['normal'] = np.transpose(tmpNormal, (2,0,1))
        return image, label


