import numpy as np

def getLoss(refImg, comparisons):
    '''
        debug comparison
    '''
    numTrue = 0
    numFalse = 0
    numTotal = comparisons.shape[0]
    print numTotal
    #refImg = np.mean(refImg, axis=2)
    delta = 0.1
    margin = 0.0
    numHeight, numWidth = refImg.shape
    for item in comparisons:
        x1, y1, x2, y2, darker = item[0:5]
        point_1 = refImg[int(round(y1*numHeight)), int(round(x1*numWidth))]
        point_2 = refImg[int(round(y2*numHeight)), int(round(x2*numWidth))]

        ratio = point_1/(point_2 + 1e-6)
        if darker == 1:
           border = 1/(1 + delta + margin)
           if (ratio  < border).data[0]:
               numTrue += 1
           else:
               numFalse += 1
        elif darker == 2:
           border = 1 + delta + margin
           if (ratio > border):
               numTrue += 1
           else:
               numFalse += 1
        elif darker == 0:
            border = 1 + delta - margin
            if ratio < border and ratio > 1.0/border:
                numTrue += 1
            else:
                numFalse += 1
    return numTrue, numFalse, numTotal
