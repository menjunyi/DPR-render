import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class loss_SAW_1(nn.Module):
    '''
        define loss_saw_1, boundary of the shadow
    '''
    def __init__(self):
        super(loss_SAW_1, self).__init__()
        self.eps = 1e-6

    def forward(self, shading, image, mask, numMask):
        '''
            give shading, mask and number of mask
            get the loss for boundary of shadow
        '''
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        log_shading = torch.log(shading + self.eps)
        log_target = torch.log(image + self.eps)
        num = shading.shape[0]*shading.shape[1]

        for i in range(1, numMask+1):
            log_diff = log_shading - log_target
            new_mask = (mask == i).float().cuda()
            masked_log_diff = torch.mul(new_mask, log_diff)
            N = torch.sum(new_mask)*num
            s1 = torch.sum( torch.pow(masked_log_diff,2))/(N  + self.eps)
            s2 = torch.pow(torch.sum(masked_log_diff),2)/(N*N + self.eps)
            total_loss += (s1 - s2)
        return total_loss/(numMask + self.eps)


class loss_SAW_2(nn.Module):
    '''
        define loss_saw_2, smooth region of shading
    '''
    def __init__(self):
        super(loss_SAW_2, self).__init__()
        self.eps = 1e-6

    def forward(self, shading, mask, numMask):
        '''
            give shading, mask and number of mask
            get the loss for smooth region of shading
        '''
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        num = shading.shape[0]*shading.shape[1]

        for i in range(1, numMask+1):
            new_mask = (mask == i).float().cuda()
            masked_shading = torch.mul(new_mask, shading)
            N = torch.sum(new_mask)*num
            s1 = torch.sum( torch.pow(masked_shading,2))/(N + self.eps) 
            s2 = torch.pow(torch.sum(masked_shading),2)/(N*N + self.eps)
            total_loss += (s1 - s2)
        return total_loss/(numMask + self.eps)
