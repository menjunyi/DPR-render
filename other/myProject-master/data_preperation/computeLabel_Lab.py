import numpy as np

def computeLabel(refImg, pointList):
    '''
        given an image and sampled data, compute the 
        relative darkness
        refImg: reflectance image (color)
        pointList: nx2 matrix, each row representing x (horizontal) and y (vertical) coordinates
    '''
    delta = 0.1
    # we have a margin, so discard some points that lies on the boundary
    boundary_1 = 1 + delta
    boundary_2 = 1 + delta

    numHeight, numWidth = refImg.shape
    meanRef = refImg.ravel()
    # shift so arr_index[:,0] contains vertical and arr_index[:,1] contains horizontal
    arr_index = pointList[:,::-1]
    # transpose so pointList[0,:] contains  first dimention and pointList[1,:] contrains second dimension
    arr_index = np.transpose(arr_index, (1,0))
    refIndex = np.ravel_multi_index(arr_index, (numHeight, numWidth))
    refValue = meanRef[refIndex]

    # compute the ratio
    numValue = refValue.shape[0]
    XX = np.tile(refValue[...,None], (1, numValue))
    YY = np.transpose(XX, (1,0))
    ratio = XX/(YY + 1e-6)
    
    ind_lighter = ratio > boundary_1
    ind_darker = ratio < 1/(boundary_1)
    ind_equal = np.logical_and(ratio < boundary_2, ratio > 1/(boundary_2))

    ind_images = np.zeros((numValue, numValue))
    ind_images = ind_lighter*2 + ind_darker*1 + ind_equal*3

    return ind_images

def sampleLabel(indexImg, mask):
    '''
        sampling point from superpixel
        sample one point from each superpixel that is not in the mask
    '''
    # skip 0
    indexImg = indexImg + 1
    # mask out invalid region
    indexImg = indexImg * mask

    indexList = np.unique(indexImg)
    selectPoints = []
    for index in indexList:
        if index != 0:
            point = np.nonzero(indexImg == index)
            numPoints = point[0].shape[0]
            if numPoints != 0:
                # select one point
                ind = np.random.choice(numPoints)
                selectPoints.append([point[1][ind], point[0][ind]])
    return np.array(selectPoints).astype(np.int32)
        
def recordLabel(refImage, indexImg, maskImg, saveName):
    '''
        refImg: reflectance image
        indexImg: index image for superpixel, nxm
        maskImg: mask for valid image region, nxm
        saveName: save the compare into a file
    '''
    numHeight, numWidth = refImage.shape
    # sample one pixel from each region
    points = sampleLabel(indexImg, maskImg)
    ind_images = computeLabel(refImage, points)
    points = points.astype(np.float32)

    fid = open(saveName, 'w')
    print>>fid, 'point1ID,point2ID,point1_x,point1_y,point2_x,point2_y,label,score'
    numPoints = len(points)
    for i in range(numPoints-1):
        for j in range(i+1, numPoints):
            if ind_images[i,j] == 1:
                print>>fid, '0,0,%0.6f,%0.6f,%0.6f,%0.6f,%d,%0.6f' % (points[i][0]/numWidth, points[i][1]/numHeight,
                        points[j][0]/numWidth, points[j][1]/numHeight, 1, 1)
            if ind_images[i,j] == 2:
                print>>fid, '0,0,%0.6f,%0.6f,%0.6f,%0.6f,%d,%0.6f' % (points[i][0]/numWidth, points[i][1]/numHeight,
                        points[j][0]/numWidth, points[j][1]/numHeight, 2, 1)
            if ind_images[i,j] == 3:
                print>>fid, '0,0,%0.6f,%0.6f,%0.6f,%0.6f,%d,%0.6f' % (points[i][0]/numWidth, points[i][1]/numHeight,
                        points[j][0]/numWidth, points[j][1]/numHeight, 0, 1)
    fid.close()
