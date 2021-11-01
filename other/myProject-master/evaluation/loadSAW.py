'''
    load data for SAW data set
    Based on CGI code
'''
import numpy as np
import torch
import os
from torch.autograd import Variable
import json
import saw_utils
from scipy.ndimage.filters import maximum_filter
#import matplotlib.pyplot as plt
import skimage
from scipy.ndimage.measurements import label
from skimage.transform import resize
import cv2
import torch.nn.functional as F
from help_saw import *

IMAGE_SIZE=128


def decompose(inputs, my_network_coarse, my_network, shadingLayer_coarse, shadingLayer):
    coarse_albedo, coarse_normal, coarse_lighting = \
            my_network_coarse(F.upsample(inputs, size=[64, 64], mode='bilinear'))
    coarse_shading = shadingLayer_coarse(coarse_normal, coarse_lighting)
    coarse_albedo = F.upsample(coarse_albedo, size=[IMAGE_SIZE, IMAGE_SIZE], mode='bilinear')
    coarse_normal = F.upsample(coarse_normal, size=[IMAGE_SIZE, IMAGE_SIZE], mode='bilinear')
    coarse_shading = F.upsample(coarse_shading, size=[IMAGE_SIZE, IMAGE_SIZE], mode='bilinear')

    # NOTE: we have a bug in coarse network for lighting, correct it
    coarse_lighting = Variable(coarse_lighting[:,0:27].data).float()
    coarse_lighting = coarse_lighting.unsqueeze(-1).unsqueeze(-1)

    # concatenate images, albedo, normal, shading as input
    inputs_albedo = torch.cat((inputs, coarse_albedo), dim=1)
    inputs_normal = torch.cat((inputs, coarse_normal), dim=1)
    inputs_lighting = torch.cat((inputs, coarse_albedo, coarse_normal, coarse_shading), dim=1)

    # concatenate images, albedo, normal, shading as input
    output_albedo, output_normal, output_lighting = my_network(inputs_albedo, inputs_normal, inputs_lighting)
    print output_lighting.shape
    print coarse_lighting.shape
    true_lighting = output_lighting + coarse_lighting.expand(-1,-1, IMAGE_SIZE, IMAGE_SIZE)
    # get shading
    output_shading = shadingLayer(F.normalize(coarse_normal + output_normal, p=2, dim=1), true_lighting) 
    output_albedo = output_albedo + coarse_albedo
    return output_albedo, output_shading
    

def compute_pr(pixel_labels_dir, splits_dir, dataset_split, class_weights, bl_filter_size, img_dir, network, thres_count=400):
    thres_list = saw_utils.gen_pr_thres_list(thres_count) 
    photo_ids = saw_utils.load_photo_ids_for_split(splits_dir=splits_dir, dataset_split=dataset_split) 
    plot_arrs = []
    line_names = []
    fn = 'pr-%s' % {'R': 'train', 'V': 'val', 'E': 'test'}[dataset_split]                                                  
    title = '%s Precision-Recall' % (
            {'R': 'Training', 'V': 'Validation', 'E': 'Test'}[dataset_split],) 
    print("FN ", fn)
    print("title ", title)
    # compute PR 
    rdic_list = get_precision_recall_list_new(pixel_labels_dir=pixel_labels_dir, thres_list=thres_list,
        photo_ids=photo_ids, class_weights=class_weights, bl_filter_size = bl_filter_size, img_dir=img_dir, network = network)

    plot_arr = np.empty((len(rdic_list) + 2, 2))

    # extrapolate starting point 
    plot_arr[0, 0] = 0.0
    plot_arr[0, 1] = rdic_list[0]['overall_prec']

    for i, rdic in enumerate(rdic_list):
        plot_arr[i+1, 0] = rdic['overall_recall']
        plot_arr[i+1, 1] = rdic['overall_prec']

    # extrapolate end point
    plot_arr[-1, 0] = 1
    plot_arr[-1, 1] = 0.5

    AP = np.trapz(plot_arr[:,1], plot_arr[:,0])

    return AP


def get_precision_recall_list_new(pixel_labels_dir, thres_list, photo_ids,
                              class_weights, bl_filter_size, img_dir, network):

    output_count = len(thres_list)
    overall_conf_mx_list = [
        np.zeros((3, 2), dtype=int)
        for _ in xrange(output_count)
    ]

    count = 0 
    total_num_img = len(photo_ids)

    for photo_id in (photo_ids):
        print("photo_id ", count, photo_id, total_num_img)
        # load photo using photo id, hdf5 format 
        print img_dir
        img_path = img_dir + str(photo_id) + ".png"

        saw_img = saw_utils.load_img_arr(img_path)
        original_h, original_w = saw_img.shape[0], saw_img.shape[1]
        saw_img = cv2.resize(saw_img, (128, 128), interpolation=cv2.INTER_CUBIC)
        #saw_img = saw_utils.resize_img_arr(saw_img)

        saw_img = np.transpose(saw_img, (2,0,1))
        input_ = torch.from_numpy(saw_img).unsqueeze(0).contiguous().float()
        input_images = Variable(input_.cuda() , requires_grad = False)

        # run model on the image to get predicted shading 
        # prediction_S , rgb_s = self.netS.forward(input_images)
        #---------------------------------------------------------
        # change to our own model
        prediction_R, prediction_S = decompose(input_images, network[0], network[1], network[2], network[3])
        #---------------------------------------------------------

        # output_path = root + '/phoenix/S6/zl548/SAW/prediction/' + str(photo_id) + ".png.h5"
        #prediction_Sr = torch.exp(prediction_S)
        prediction_Sr = prediction_S
        # prediction_Sr = torch.pow(prediction_Sr, 0.4545)
        prediction_S_np = prediction_Sr.data[0,0,:,:].cpu().numpy() 
        prediction_S_np = resize(prediction_S_np, (original_h, original_w), order=1, preserve_range=True)

        # compute confusion matrix
        conf_mx_list = eval_on_images( shading_image_arr = prediction_S_np,
            pixel_labels_dir=pixel_labels_dir, thres_list=thres_list,
            photo_id=photo_id, bl_filter_size = bl_filter_size, img_dir=img_dir
        )

        for i, conf_mx in enumerate(conf_mx_list):
            # If this image didn't have any labels
            if conf_mx is None:
                continue
            overall_conf_mx_list[i] += conf_mx

        count += 1

        ret = []
        for i in xrange(output_count):
            overall_prec, overall_recall = saw_utils.get_pr_from_conf_mx(
                conf_mx=overall_conf_mx_list[i], class_weights=class_weights,
            )

            ret.append(dict(
                overall_prec=overall_prec,
                overall_recall=overall_recall,
                overall_conf_mx=overall_conf_mx_list[i],
            ))
    return ret

def eval_on_images(shading_image_arr, pixel_labels_dir, thres_list, photo_id, bl_filter_size, img_dir):
    """
    This method generates a list of precision-recall pairs and confusion
    matrices for each threshold provided in ``thres_list`` for a specific
    photo.

    :param shading_image_arr: predicted shading images

    :param pixel_labels_dir: Directory which contains the SAW pixel labels for each photo.

    :param thres_list: List of shading gradient magnitude thresholds we use to
    generate points on the precision-recall curve.

    :param photo_id: ID of the photo we want to evaluate on.

    :param bl_filter_size: The size of the maximum filter used on the shading
    gradient magnitude image. We used 10 in the paper. If 0, we do not filter.
    """

    shading_image_linear_grayscale = shading_image_arr
    shading_image_linear_grayscale[shading_image_linear_grayscale < 1e-4] = 1e-4
    shading_image_linear_grayscale = np.log(shading_image_linear_grayscale)

    shading_gradmag = saw_utils.compute_gradmag(shading_image_linear_grayscale)
    shading_gradmag = np.abs(shading_gradmag)

    if bl_filter_size:
        shading_gradmag_max = maximum_filter(shading_gradmag, size=bl_filter_size)

    # We have the following ground truth labels:
    # (0) normal/depth discontinuity non-smooth shading (NS-ND)
    # (1) shadow boundary non-smooth shading (NS-SB)
    # (2) smooth shading (S)
    # (100) no data, ignored
    y_true = saw_utils.load_pixel_labels(pixel_labels_dir=pixel_labels_dir, photo_id=photo_id)
    
    img_path = img_dir+ str(photo_id) + ".png"

    # diffuclut and harder dataset
    srgb_img = saw_utils.load_img_arr(img_path)
    srgb_img = np.mean(srgb_img, axis = 2)
    img_gradmag = saw_utils.compute_gradmag(srgb_img)

    smooth_mask = (y_true == 2)
    average_gradient = np.zeros_like(img_gradmag)
    # find every connected component
    labeled_array, num_features = label(smooth_mask)
    for j in range(1, num_features+1):
        # for each connected component, compute the average image graident for the region
        avg = np.mean(img_gradmag[labeled_array == j])
        average_gradient[labeled_array == j]  = avg

    average_gradient = np.ravel(average_gradient)

    y_true = np.ravel(y_true)
    ignored_mask = y_true > 99

    # If we don't have labels for this photo (so everything is ignored), return
    # None
    if np.all(ignored_mask):
        return [None] * len(thres_list)

    ret = []
    for thres in thres_list:
        y_pred = (shading_gradmag < thres).astype(int)
        y_pred_max = (shading_gradmag_max < thres).astype(int)
        y_pred = np.ravel(y_pred)
        y_pred_max = np.ravel(y_pred_max)
        # Note: y_pred should have the same image resolution as y_true
        assert y_pred.shape == y_true.shape

        # confusion_matrix = saw_utils.grouped_confusion_matrix(y_true[~ignored_mask], y_pred[~ignored_mask], y_pred_max[~ignored_mask])
        confusion_matrix = saw_utils.grouped_weighted_confusion_matrix(y_true[~ignored_mask], y_pred[~ignored_mask], y_pred_max[~ignored_mask], average_gradient[~ignored_mask])
        ret.append(confusion_matrix)

    return ret
