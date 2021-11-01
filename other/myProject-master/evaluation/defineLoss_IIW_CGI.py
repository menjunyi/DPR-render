import torch
import numpy as np
from torch.autograd import Variable
import cv2
from skimage.transform import resize
import json

def evlaute_iiw(self, input_, targets):
    # switch to evaluation mode
    input_images = Variable(input_.cuda() , requires_grad = False)
    prediction_S, prediction_R , rgb_s = self.netG.forward(input_images)

    prediction_R = torch.exp(prediction_R)

    return self.evaluate_WHDR(prediction_R, targets)

def compute_whdr(reflectance, judgements, delta=0.1):
    '''
        evalute whdr for one single image
    '''
    points = judgements['intrinsic_points']
    comparisons = judgements['intrinsic_comparisons']
    id_to_points = {p['id']: p for p in points}
    rows, cols = reflectance.shape[0:2]

    error_sum = 0.0
    weight_sum = 0.0

    for c in comparisons:
        # "darker" is "J_i" in our paper
        darker = c['darker']
        if darker not in ('1', '2', 'E'):
            continue

        # "darker_score" is "w_i" in our paper
        weight = c['darker_score']
        if weight <= 0.0 or weight is None:
            continue

        point1 = id_to_points[c['point1']]
        point2 = id_to_points[c['point2']]
        if not point1['opaque'] or not point2['opaque']:
            continue

        # convert to grayscale and threshold
        l1 = max(1e-10, np.mean(reflectance[
            int(point1['y'] * rows), int(point1['x'] * cols), ...]))
        l2 = max(1e-10, np.mean(reflectance[
            int(point2['y'] * rows), int(point2['x'] * cols), ...]))

        # # convert algorithm value to the same units as human judgements
        if l2 / l1 > 1.0 + delta:
            alg_darker = '1'
        elif l1 / l2 > 1.0 + delta:
            alg_darker = '2'
        else:
            alg_darker = 'E'


        if darker != alg_darker:
            error_sum += weight

        weight_sum += weight

    if weight_sum:
        return (error_sum / weight_sum)
    else:
        return None

def evaluate_WHDR(prediction_R, targets):
    '''
        evalaute WHDR for a batch
    '''
    total_whdr = float(0)
    count = float(0) 

    for i in range(0, prediction_R.size(0)):
        print(targets['path'][i])
        prediction_R_np = prediction_R.data[i,:,:,:].cpu().numpy()
        prediction_R_np = np.transpose(prediction_R_np, (1,2,0))

        #o_h = targets['oringinal_shape'][0].numpy()
        #o_w = targets['oringinal_shape'][1].numpy()
        ## resize to original resolution 
        #prediction_R_np = resize(prediction_R_np, (o_h[i],o_w[i]), order=1, preserve_range=True)
        prediction_R_np = resize(prediction_R_np, (256,256), order=1, preserve_range=True)

        # load Json judgement 
        judgements = json.load(open(targets["judgements_path"][i]))
        whdr = compute_whdr(prediction_R_np, judgements, 0.1)

        total_whdr += whdr
        count += 1.

    return total_whdr, count
