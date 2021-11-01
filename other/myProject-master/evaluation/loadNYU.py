import os
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import numpy as np
import random
random.seed(0)
import cv2

class NYU(Dataset):
    '''
    	loading IIW data sets
    '''
    
    def __init__(self, dataFolder, normalFolder, maskFolder, fileListName, transform=None):
        '''
        	dataFolder: contains images
        	normalFolder: contains normal information
            maskFolder: contains mask 
        	fileListName: all file names
        '''
        
        self.fileList = []
        with open(fileListName) as f:
            for line in f:
                tmp = line.strip().split('/')[1]
                self.fileList.append(tmp)
        
        self.dataFolder = dataFolder
        self.normalFolder = normalFolder
        self.maskFolder = maskFolder
        self.transform = transform
    
    def __len__(self):
        return len(self.fileList)
    
    def __getitem__(self, idx):
        fileName = self.fileList[idx]
        #print fileName
        imgName = os.path.join(self.dataFolder, fileName + '.png')
        image = io.imread(imgName)
        if len(image.shape)==2:
            image = np.tile(image[...,None], (1, 3))
        
        normalName = os.path.join(self.normalFolder, fileName + '_normal.png')
        normal = io.imread(normalName)
        
        maskName = os.path.join(self.maskFolder, fileName + '_mask.png')
        mask = io.imread(maskName)
        
        if self.transform:
            image, normal, mask = \
                    self.transform([image, normal, mask])
        return image, normal, mask

class testTransfer(object):
    def __init__(self, output_size=64):
        # we need to think about this latter
        self.size=output_size
    def __call__(self, sample):
        # center crop
        image, normal, mask = sample

        # directly resize the image
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        normal = cv2.resize(normal, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        
        image = 1.0*image/255.0
        normal = normal.astype(np.float)
        normal = (normal/255.0-0.5)*2
        normal = normal/(np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6)
        mask = 1.0*mask/255.0
        mask = mask[...,None]
        
        return image, normal, mask


class cropImg(object):
    '''
        expand image first and then crop
    '''
    def __init__(self, output_size=224, expand=10):
        self.size = output_size
        self.expand = 10
    def __call__(self, sample):
        image, normal, mask = sample
        image = cv2.resize(image, (self.size + self.expand, self.size + self.expand), interpolation=cv2.INTER_CUBIC)
        normal = cv2.resize(normal, (self.size + self.expand, self.size + self.expand), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (self.size + self.expand, self.size + self.expand), interpolation=cv2.INTER_CUBIC)
        
        # random crop
        H = image.shape[0]
        W = image.shape[1]
        maxH = H - self.size
        maxW = W - self.size
        sH = random.randint(0, maxH)
        sW = random.randint(0, maxW)
        
        image = image[sH:sH+self.size, sW:sW+self.size,:]
        normal = normal[sH:sH+self.size, sW:sW+self.size,:]
        mask = mask[sH:sH+self.size, sW:sW+self.size]
        
        # convert to 0-1
        image = 1.0*image/255.0
        normal = normal.astype(np.float)
        normal = (normal/255.0 - 0.5)*2
        normal = normal/(np.tile(np.linalg.norm(normal, axis=-1, keepdims=True), (1,1,3)) + 1e-6)
        
        mask = 1.0*mask/255.0
        ind = np.abs(mask - 1) < 1e-3
        mask[ind] = 1
        ind = not ind
        mask[ind] = 0
        return image, normal, mask


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, normal, mask = sample
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        normal = normal.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        return torch.from_numpy(image), \
            torch.from_numpy(normal), \
            torch.from_numpy(mask)
