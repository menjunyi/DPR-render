import pandas as pd
from torch.autograd import Variable
import numpy as np
import torch
import time
import torch.nn as nn

class evaluateNetwork(object):
    '''
        compute loss
    '''
    def __init__(self, criterion_mask, criterion_WHDR, criterion_WHDR_paper, criterion_SAW_1, criterion_SAW_2, gradientLayer): 
        # we may use three criterions:
        # 1. mask loss (including SiMSE loss)
        # 2. WHDR loss (loss for albedo)
        # 3. SAW loss (loss for shading)
        self.criterion_mask = criterion_mask
        self.criterion_WHDR = criterion_WHDR
        self.criterion_WHDR_paper = criterion_WHDR_paper
        self.criterion_SAW_1 = criterion_SAW_1
        self.criterion_SAW_2 = criterion_SAW_2
        self.gradientLayer = gradientLayer
        self.eps = 1e-6
        self.SUNCG_list = ['albedo', 'normal', 'shading']
        self.numCompare = 300 # maximum number of comparison for training and testing
        self.IIW_random = 0.8 # each time select 80 % of comparisons to add randomness to gradient

    def getLoss_ABS(self, source, target, mask):
        '''
            return the mask loss with abs
        '''
        loss = self.criterion_mask(mask, torch.abs(source - target))
        return loss

    def getLoss_L2(self, source, target):
        '''
            return the l2 loss
            for reconstruction
        '''
        loss = torch.sqrt(torch.mean((source - target)**2))
        return loss

    def getLoss_WHDR(self, source, label):
        '''
            get WHDR loss
        '''
        loss = self.criterion_WHDR(source, label)
        return loss

    def getLoss_WHDR_paper(self, source, label):
        '''
            get WHDR loss
        '''
        loss = self.criterion_WHDR_paper(source, label)
        return loss

    def getLoss_SAW_1(self, source, image, mask, numLabel):
        '''
            get SAW loss 1, loss for boundary of shadow
        '''
        loss = self.criterion_SAW_1(source, image, mask, numLabel)
        return loss

    def getLoss_SAW_2(self, source, mask, numLabel):
        '''
            get SAW loss 2, loss for smooth region of shading
        '''
        loss = self.criterion_SAW_2(source, mask, numLabel)
        return loss


    def getLoss_lighting(self, output_lighting):
        pass

    def getAlpha(self, source, target):
        '''
            compute a alpha to minimize
            (alpha*source - target)^2 
        '''
        if source.size()[0] != target.size()[0] or \
        	source.size()[1] != target.size()[1] or \
        	source.size()[2] != target.size()[2] or \
        	source.size()[3] != target.size()[3]:
                raise ValueError('size of groud truth and ouptut does not match')
        numImages = source.shape[0]
        source = source.view(numImages, -1)
        target = source.view(numImages, -1)
                    
        alpha = torch.sum(target*source,dim=1)/(torch.sum(source**2, dim=1) + self.eps)
        return alpha

    def loadLabel(self, labelName, dataName, numHeight, numWidth, indType):
        '''
            load csv file 
        '''
        data_csv = pd.read_csv(labelName).values
        data_csv = data_csv[:,2:]
        data_csv[:,0] = np.round(data_csv[:,0]*numWidth)
        data_csv[:,2] = np.round(data_csv[:,2]*numHeight)
        data_csv[:,1] = np.round(data_csv[:,1]*numWidth)
        data_csv[:,3] = np.round(data_csv[:,3]*numHeight)

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
        elif dataName == 'IIW' and indType == 'train':
            numSelected = int(data_csv.shape[0]*self.IIW_random)

            if numSelected > 10:
                # at least 10 comparisons should be selected
                np.random.shuffle(data_csv)
                data_csv = data_csv[0:numSelected]
        return data_csv

    def IIW_loss(self, sourceData, targetData, dataName, scale_factor=1, trainType='train'):
        '''
            compute IIW loss
        '''
        loss = {}

        upSample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        #if dataName == 'SUNCG':
        #    tmp = upSample(sourceData)
        #else:
        #    tmp = upSample(sourceData['albedo'])
        tmp = upSample(sourceData['albedo'])
        numHeight = int(tmp.shape[2]) - 1 
        numWidth = int(tmp.shape[3]) - 1
        # loss of WHDR
        WHDR_labelName = targetData['label']
        WHDR_loss = Variable(torch.cuda.FloatTensor([0]))
        # load label and compute the loss
        for (i, labelName) in enumerate(WHDR_labelName):
            label = Variable(torch.from_numpy(self.loadLabel(labelName, dataName, numHeight, numWidth, trainType)).cuda()).float()
            if trainType == 'train':
                WHDR_loss += self.getLoss_WHDR(tmp[i], label)
            else:
                WHDR_loss += self.getLoss_WHDR_paper(tmp[i], label)
        loss['WHDR'] = WHDR_loss/len(WHDR_labelName)
        return loss

    def IIW_colorLoss(self, sourceData, imageData, targetData):
        '''
            compute color loss for albedo of real images
            and reconstruction loss
        '''
        loss = {}

        #mask = targetData['mask']
        #imgMask = mask.expand(-1, 3, -1, -1)
        #gradMask = mask.expand(-1, 6, -1, -1)
        # no mask for IIW
        numSamples = imageData.shape[0]
        numHeight = imageData.shape[2]
        numWidth = imageData.shape[3]
        imgMask = Variable(torch.ones(numSamples, 3, numHeight, numWidth).cuda()).float()
        gradMask = Variable(torch.ones(numSamples, 6, numHeight, numWidth).cuda()).float()

        output_albedo = sourceData['albedo']
        output_color = output_albedo/(torch.sum(output_albedo, dim=1, keepdim=True) + 1e-6).repeat(1,3,1,1)
        grad_color = self.gradientLayer.forward(output_color)
        gt_color = imageData/(torch.sum(imageData, dim=1, keepdim=True) + 1e-6).repeat(1,3,1,1)
        grad_gt_color = self.gradientLayer.forward(gt_color)

        loss['color'] = self.getLoss_ABS(output_color, gt_color, imgMask)
        loss['color_grad'] = self.getLoss_ABS(grad_color, grad_gt_color, gradMask)

        # get reconstruction loss
        output_shading = sourceData['shading']
        reconstruct_img = output_shading*output_albedo
        alpha = self.getAlpha(reconstruct_img, imageData)
 
        # ugly implementation
        numImages, numChannels, numRows, numCols = reconstruct_img.shape
        tmp_alpha = alpha.view(numImages, 1).repeat(1, numChannels, 
                numRows, numCols)
        tmp_alpha_1 = tmp_alpha.view(numImages, numChannels, numRows, numCols)
        alphaImages = tmp_alpha_1*reconstruct_img
        loss['reconstruct'] = self.getLoss_ABS(alphaImages, imageData, imgMask)
        alphaImages_grad = self.gradientLayer.forward(alphaImages)
        images_grad = self.gradientLayer.forward(imageData)
        loss['reconstruct_grad'] = self.getLoss_ABS(alphaImages_grad, images_grad, gradMask)
        
        return loss

        

    def SUNCG_data(self, sourceData, targetData):
        '''
            SUNCG data contains:
            mask loss for albedo, shading, normal
            WHDR loss for albedo
        '''
        # use l1 loss
        mask = targetData['mask']
        imgMask = mask.expand(-1, 3, -1, -1)
        gradMask = mask.expand(-1, 6, -1, -1)
        loss = {}

        for item in self.SUNCG_list:
            source = sourceData[item]
            target = targetData[item]
            loss[item] = self.getLoss_ABS(source, target, imgMask)
            source_grad = self.gradientLayer.forward(source)
            target_grad = self.gradientLayer.forward(target)
            loss[item + '_grad'] = self.getLoss_ABS(source_grad, target_grad, gradMask)

        return  loss


    def IIW_data(self, sourceData, images, targetData):
        '''
            IIW data contains:
            WHDR loss for albedo
            SAW loss for shading
        '''
        loss = {}

        # SAW loss
        saw_loss_1 = Variable(torch.cuda.FloatTensor([0]))
        saw_loss_2 = Variable(torch.cuda.FloatTensor([0]))
        total_num = sourceData['shading'].shape[0]
        for i in range(total_num):
            shading = sourceData['shading'][i]
            saw_loss_1 += self.getLoss_SAW_1(shading, images, targetData['saw_mask_1'][i], 
                    targetData['num_saw_mask_1'][i])
            saw_loss_2 += self.getLoss_SAW_2(shading, targetData['saw_mask_2'][i],
                    targetData['num_saw_mask_2'][i])
        loss['saw_loss_1'] = saw_loss_1/total_num
        loss['saw_loss_2'] = saw_loss_2/total_num

        # reconstruction loss
        reconstruct_img = sourceData['albedo']*sourceData['shading']
        alpha = self.getAlpha(reconstruct_img, images)
 
        # ugly implementation
        numImages, numChannels, numRows, numCols = reconstruct_img.shape
        tmp_alpha = alpha.view(numImages, 1).repeat(1, numChannels, 
                numRows, numCols)
        tmp_alpha_1 = tmp_alpha.view(numImages, numChannels, numRows, numCols)
        alphaImages = tmp_alpha_1*reconstruct_img
        loss['reconstruct'] = self.getLoss_L2(alphaImages, images)
        return loss

    def NYU_data(self, sourceData, images, targetData):
        '''
            NYU data contains:
            SAW loss for shading
            mask loss for normal
        '''
        loss = {}
        mask = targetData['mask']
        # loss for normal
        imgMask = mask.expand(-1, 3, -1, -1)
        gradMask = mask.expand(-1, 6, -1, -1)
        source = sourceData['normal']
        target = targetData['normal']
        loss['normal'] = self.getLoss_ABS(source, target, imgMask)
        source_grad = self.gradientLayer.forward(source)
        target_grad = self.gradientLayer.forward(target)
        loss['normal_grad'] = self.getLoss_ABS(source_grad, target_grad, gradMask)

        # SAW loss
        saw_loss_1 = Variable(torch.cuda.FloatTensor([0]))
        saw_loss_2 = Variable(torch.cuda.FloatTensor([0]))
        total_num = sourceData['shading'].shape[0]
        for i in range(total_num):
            shading = sourceData['shading'][i]
            saw_loss_1 += self.getLoss_SAW_1(shading, images, targetData['saw_mask_1'][i], 
                    targetData['num_saw_mask_1'][i])
            saw_loss_2 += self.getLoss_SAW_2(shading, targetData['saw_mask_2'][i],
                    targetData['num_saw_mask_2'][i])
        loss['saw_loss_1'] = saw_loss_1/total_num
        loss['saw_loss_2'] = saw_loss_2/total_num

        # reconstruction loss
        reconstruct_img = sourceData['albedo']*sourceData['shading']
        alpha = self.getAlpha(reconstruct_img, images)

        # ugly implementation
        numImages, numChannels, numRows, numCols = reconstruct_img.shape
        tmp_alpha = alpha.view(numImages, 1).repeat(1, numChannels, 
                numRows, numCols)
        tmp_alpha_1 = tmp_alpha.view(numImages, numChannels, numRows, numCols)
        alphaImages = tmp_alpha_1*reconstruct_img
        loss['reconstruct'] = self.getLoss_L2(alphaImages, images)
        return loss
