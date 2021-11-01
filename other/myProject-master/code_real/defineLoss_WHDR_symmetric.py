import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


'''
    according to Xiang's suggestion, we define the loss to be symmetric
'''

# senital for divided by 0
eps = 1e-10

class WhdrHingeLoss(nn.Module):
    def __init__(self, margin=0.05, delta=0.1):
        super(WhdrHingeLoss, self).__init__()
        self.margin = margin
        self.delta = delta
        self.helpRGB = Variable(torch.cuda.FloatTensor([0.212671, 0.715160, 0.072169]))

    def rgb2lab(self, data):
        numHeight = data.shape[1]
        numWidth = data.shape[2]

        #----------------------------------------------------------------
        # linearize RGB according to WIKI sRGB
        ind_1 = (data <= 0.04045).float()
        ind_2 = (data > 0.04045).float()
        data = (data/12.92)*ind_1 + ((data + 0.055)/(1+0.055))**2.4*ind_2
        # ----------------------------------------------------------------

        v_input = torch.matmul(self.helpRGB, data.view(3,-1))
        v_input = v_input.view(numHeight, numWidth)
        ind_1 = (v_input > 0.008856).float()
        ind_2 = (v_input <= 0.008856).float()
        v_input = (116.0*v_input**(1.0/3.0)-16.0)*ind_1 + \
                903.3*v_input*ind_2

        # normalize pixel value to 0 to 1
        v_input = (v_input + 1)/100.0
        return v_input
    
    def forward(self, v_input, comparisons):
        # record loss
        #total_loss = Variable(torch.cuda.FloatTensor([0]))
        v_input = torch.sum(v_input, dim=0)/3
        #v_input = self.rgb2lab(v_input)
        # rescale images to avoid 0 
        v_input = (v_input*255.0+1)/256.0
        loss = Variable(torch.cuda.FloatTensor([0]))
        total_weight = 1e-6
        numComparisons = comparisons.shape[0]
        for (i, comparison) in enumerate(comparisons):
            item = list(comparison.cpu().data.numpy())
            x1, y1, x2, y2, darker = item[0:5]
            weight = float(item[5])
            total_weight += weight
            
            r_1 = v_input[y1, x1]
            r_2 = v_input[y2, x2]
            
            ratio = r_1 / (r_2 + eps)
            ratio_inv = r_2 / (r_1 + eps)

            if darker  == 1: 
                # r_1 is darker than r_2
                border = 1/(1 + self.delta + self.margin)
                if (ratio  > border).data[0]:
                    loss = loss + weight*(ratio - border)
                    # to be symmetric, we also add this loss
                    loss = loss + weight*((1 + self.delta + self.margin) - ratio_inv)
            elif darker == 2:
                # r_2 is darker than r_1
                border = 1 + self.delta + self.margin
                if (ratio < border).data[0]:
                    loss = loss + weight*(border - ratio)
                    loss = loss + weight*(ratio_inv - 1/(1+self.delta + self.margin))
            elif darker == 0:
                # r_1 and r_2 are more or less the same
                if self.margin <= self.delta:
                    border_right = 1 + self.delta - self.margin
                    border_left = 1/border_right
                    # loss = max(0, border_left - y, y - border_right)
                    if (ratio > border_right).data[0]:
                        loss = loss + weight*(ratio - border_right)
                        loss = loss + weight*(border_left - ratio_inv)
                    else:
                        if (ratio < border_left).data[0]:
                            loss = loss + weight*(border_left - ratio)
                            loss = loss + weight*(ratio_inv - border_right)
                else:
                    border = 1 + self.delta - self.margin
                    loss = loss + max(1/border-y, y-border)
            else:
            		raise Exception('darker is neighter E(0), 1 or 2')
        loss = loss/numComparisons
        #loss = loss/total_weight
        return loss

class WhdrTestLoss_Paper(nn.Module):
    def __init__(self, delta=0.1):
        super(WhdrTestLoss_Paper, self).__init__()
        self.delta = delta
        self.helpRGB = Variable(torch.cuda.FloatTensor([0.212671, 0.715160, 0.072169]))
    def rgb2lab(self, data):
        numHeight = data.shape[1]
        numWidth = data.shape[2]

        v_input = torch.matmul(self.helpRGB, data.view(3,-1))
        v_input = v_input.view(numHeight, numWidth)
        ind_1 = (v_input > 0.008856).float()
        ind_2 = (v_input <= 0.008856).float()
        v_input = (116.0*v_input**(1.0/3.0)-16.0)*ind_1 + \
                903.3*v_input*ind_2

        # normalize pixel value to 0 to 1
        v_input = (v_input + 1)/100.0
        return v_input
    
    def forward(self, v_input, comparisons):
        # record loss
        v_input = torch.sum(v_input, dim=0)/3
        #v_input = self.rgb2lab(v_input)
        # avoid zero
        v_input = (v_input*255.0+1)/256.0
        loss = Variable(torch.cuda.FloatTensor([0]))
        total_weight = 0
        #print comparisons[count]
        for (i, comparison) in enumerate(comparisons):
            item = list(comparison.cpu().data.numpy())
            x1, y1, x2, y2, darker = item[0:5]
            weight = float(item[5])
            total_weight += weight
            
            r_1 = v_input[y1, x1]
            r_2 = v_input[y2, x2]
            if (r_2/(r_1 + eps) > 1.0 + self.delta).data[0]:
                # darker
                alg = 1
            elif (r_1/(r_2 + eps) > 1.0 + self.delta).data[0]:
                alg = 2
            else:
                alg = 0
            if alg != darker:
                loss += weight
        
        loss = loss/total_weight
        return loss
