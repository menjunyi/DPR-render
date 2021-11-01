import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re

def drawLoss(epochs, loss, labelName, saveName):
    plt.figure()
    lineStyle = ['r-', 'b-', 'm-', 'c-', 'g-', ]
    for i in range(loss.shape[0]):
    		plt.plot(epochs, loss[i], lineStyle[i], label=labelName[i], linewidth=4)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('loss', fontsize=18)
    plt.legend(fontsize=20)
    plt.savefig(saveName)

def loadSepLoss(fileName):
    allLoss_albedo = []
    allLoss_shading = []
    allLoss_normal = []
    allLoss_lighting = []
    count = 0
    with open(fileName) as f:
        for line in f:
            if count != 0:
                tmp = line.strip()
                tmp = tmp.split()
                allLoss_albedo.append(float(tmp[0]))
                allLoss_shading.append(float(tmp[2]))
                allLoss_normal.append(float(tmp[4]))
                allLoss_lighting.append(float(tmp[5]))
            count += 1
    return allLoss_albedo, allLoss_shading, allLoss_normal, allLoss_lighting

def drawFigure(resultFolder, starting_epoch=0):
    trainingLoss_albedo, trainingLoss_shading, \
        trainingLoss_normal, trainingLoss_lighting = \
        loadSepLoss(os.path.join(resultFolder, 'training_sep.log'))
    testingLoss_albedo, testingLoss_shading, \
        testingLoss_normal, testingLoss_lighting = \
        loadSepLoss(os.path.join(resultFolder, 'testing_sep.log'))
    
    numEpochs = len(trainingLoss_albedo)
    epochs = np.arange(numEpochs)
    
    trainingLoss = []
    trainingLoss.append(trainingLoss_albedo)
    trainingLoss.append(trainingLoss_shading)
    trainingLoss.append(trainingLoss_normal)
    trainingLoss.append(trainingLoss_lighting)
    trainingLoss = np.array(trainingLoss)
    
    testingLoss = []
    testingLoss.append(testingLoss_albedo)
    testingLoss.append(testingLoss_shading)
    testingLoss.append(testingLoss_normal)
    testingLoss.append(testingLoss_lighting)
    testingLoss = np.array(testingLoss)
    labelName = ['albedo', 'shading', 'normal', 'lighting']
    drawLoss(epochs, trainingLoss, labelName, os.path.join(resultFolder, 'loss_training.png'))
    drawLoss(epochs, testingLoss, labelName, os.path.join(resultFolder, 'loss_testing.png'))

if __name__ == '__main__':
    resultFolder = sys.argv[1]
    if len(sys.argv) > 2:
        starting_epoch = int(sys.argv[2])
        drawFigure(resultFolder, starting_epoch)
    else:
        drawFigure(resultFolder)
