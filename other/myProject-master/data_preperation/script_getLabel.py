import os
import cv2
import imageio
from computeLabel import *
import pandas as pd
from debug_computeLabel import *
import time
from multiprocessing import Pool
import sys

baseFolder = '/scratch1/other_code/pbrs/render_images/'
#imgFolder = os.path.join(baseFolder, 'images_color/')
imgFolder = os.path.join(baseFolder, 'albedo/')
superPixelFolder = os.path.join(baseFolder, 'images_superPixel/')
maskFolder = os.path.join(baseFolder, 'images_mask_correct/')
saveFolder = os.path.join(baseFolder, 'pixel_order_albedo_mean/')


imgList = []
with open(os.path.join(baseFolder, 'validFile.list')) as f:
    for line in f:
        imgList.append(line.strip())
paramList = []
for (i, item) in enumerate(imgList):
    #if i > 12:
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

    refImg = cv2.imread(refName)
    refImg = 1.0*refImg/255.0

    indexImg = imageio.imread(indexName)

    maskImg = cv2.imread(maskName)
    maskImg = np.mean(maskImg/255.0, axis=2)
    maskImg = maskImg == 1
    recordLabel(refImg, indexImg, maskImg, saveName)

if __name__ == '__main__':
    # pass
    index = int(sys.argv[1])
    baseNum = 9622
    #baseNum = 2
    start = baseNum*index
    end = min(baseNum*(index+1), 57730)
    for i in range(start, end):
        wraper_compute(paramList[i])

    #print 'time used for one image is %s' %(time.time() - begin_time)

    #label_CSV = pd.read_csv(saveName)
    #label = np.zeros((label_CSV.shape[0], 6))
    #for i in range(label_CSV.shape[0]):
    #    item = label_CSV.ix[i]
    #    tmpLabel = [float(x) for x in item]
    #    tmpLabel = tmpLabel[2:]
    #    label[i] = tmpLabel
    #numTrue, numFalse, numTotal = getLoss(refImg, label)
    #totalTrue += numTrue
    #totalFalse += numFalse
    #totalNum += numTotal


#print 'ture rate is %0.4f' % (1.0*totalTrue/totalNum)
#print 'false rate is %0.4f' % (1.0*totalFalse/totalNum)
