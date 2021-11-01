import os
import cv2
import imageio
from computeLabel_Lab import *
import pandas as pd
from debug_computeLabel import *
import time
from multiprocessing import Pool
import sys
from skimage import io
from skimage import color

baseFolder = '/scratch1/other_code/pbrs/render_images/'
#imgFolder = os.path.join(baseFolder, 'images_color/')
imgFolder = os.path.join(baseFolder, 'albedo/')
superPixelFolder = os.path.join(baseFolder, 'images_superPixel/')
maskFolder = os.path.join(baseFolder, 'images_mask_correct/')
saveFolder = os.path.join(baseFolder, 'pixel_order_albedo/')

def rgb2lab(img):
    '''
        convert to Lab channel
    '''
    img = 1.0*img/255.0
    ind_1 = (img <= 0.04045)*1.0
    ind_2 = (img > 0.04045)*1.0
    img = (img/12.92)*ind_1 + ((img + 0.055)/(1+0.055))**2.4*ind_2
    helpLab = np.array([0.212671, 0.715160, 0.072169])
    numHeight = img.shape[0]
    numWidth = img.shape[1]
    img = img.transpose((2,0,1))
    tmp = np.matmul(helpLab, img.reshape(3, -1))
    tmp = tmp.reshape(numHeight,numWidth)

    ind_1 = (tmp > 0.008856)*1.0
    ind_2 = (tmp <= 0.008856)*1.0
    tmp = (116.0*tmp**(1.0/3.0)-16.0)*ind_1 + \
            903.3*tmp*ind_2
    tmp = (tmp + 1)/101.0
    return  tmp

imgList = []
with open(os.path.join(baseFolder, 'validFile.list')) as f:
    for line in f:
        imgList.append(line.strip())
paramList = []
for (i, item) in enumerate(imgList):
    #if i > 5:
    #    break
    begin_time = time.time()
    tmp = item.split('/')
    subFolder = tmp[0]
    imgName = tmp[1]
    saveSubFolder = os.path.join(saveFolder, subFolder)
    if not os.path.exists(saveSubFolder):
        os.makedirs(saveSubFolder)
    refName = os.path.join(imgFolder, item)
    indexName = os.path.join(superPixelFolder, item[0:-3] + 'pgm')
    maskName = os.path.join(maskFolder, item)
    saveName = os.path.join(saveSubFolder, tmp[1][0:-3] + 'csv')
    paramList.append([refName, indexName, maskName, saveName])


def wraper_compute(nameList):
    refName = nameList[0]
    indexName = nameList[1]
    maskName = nameList[2]
    saveName = nameList[3]

    refImg = io.imread(refName)
    refImg = rgb2lab(refImg)

    indexImg = imageio.imread(indexName)

    maskImg = cv2.imread(maskName)
    maskImg = np.mean(maskImg/255.0, axis=2)
    maskImg = maskImg == 1
    recordLabel(refImg, indexImg, maskImg, saveName)


    #label_CSV = pd.read_csv(saveName)
    #label = np.zeros((label_CSV.shape[0], 6))
    #for i in range(label_CSV.shape[0]):
    #    item = label_CSV.ix[i]
    #    tmpLabel = [float(x) for x in item]
    #    tmpLabel = tmpLabel[2:]
    #    label[i] = tmpLabel
    #numTrue, numFalse, numTotal = getLoss(refImg, label)
    #print  'numTrue is %d' % numTrue
    #print  'numFalse is %d' % numFalse
    #print  'numFalse is %d' % numTotal

if __name__ == '__main__':
    # pass
    index = int(sys.argv[1])
    #index = 0
    baseNum = 9622
    #baseNum = 2
    start = baseNum*index
    end = min(baseNum*(index+1), 57730)
    for i in range(start, end):
        wraper_compute(paramList[i])

