import sys
sys.path.append('model')
sys.path.append('utils')

from model.defineHourglass_512_gray_skip import *
import cv2
import os
import torch
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class image_gradient(nn.Module):
    def __init__(self) -> None:
        super(image_gradient, self).__init__()
        kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -2]])
        kernel_x = torch.from_numpy(kernel_x).float().unsqueeze(0).unsqueeze(0).to(device)
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        kernel_y = torch.from_numpy(kernel_y).float().unsqueeze(0).unsqueeze(0).to(device)
        self.weight_x = nn.Parameter(kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(kernel_y, requires_grad=False)
    
    def forward(self, img):
        x_grad = F.conv2d(img, self.weight_x, padding=1)
        y_grad = F.conv2d(img, self.weight_y, padding=1)
        return x_grad, y_grad

calculate_grad = image_gradient()

def image_loss(output_img, output_sh, target_img, target_sh):
    N_img = output_img.shape[-2] * output_img.shape[-1]
    l1 = nn.L1Loss(reduction='sum')(output_img, target_img)

    out_img_x_grad, out_img_y_grad = calculate_grad(output_img)
    target_img_x_grad, target_img_y_grad = calculate_grad(target_img)
    l2_x = nn.L1Loss(reduction='sum')(out_img_x_grad, target_img_x_grad)
    l2_y = nn.L1Loss(reduction='sum')(out_img_y_grad, target_img_y_grad)

    l3 = nn.MSELoss(reduction='sum')(output_sh, target_sh)
    
    return (l1 + l2_x + l2_y) / N_img + l3

def feature_loss(output_feature, other_feature):
    return nn.MSELoss(reduction='mean')(output_feature, other_feature)

hourglass_network = HourglassNet()

#hourglass_network.cuda()
hourglass_network.train(True)

trainingFolder = './RI_render_DPR'
faceList = []
with open(trainingFolder + '/data.list') as f:
    for line in f:
        faceList.append(line.strip())
relighted_imgPath =  trainingFolder + '/relighting/'

all_inputL = []
all_targetL = []
all_inputsh = []
all_targetsh = []
all_imgname = []

for item in faceList:
    imgName = item.split('.')[0]
    foldername = os.path.join(relighted_imgPath, imgName)
    lights = []

    all_targetL_for_item = []
    all_targetsh_for_item = []
    all_inputL_for_item = []
    all_inputsh_for_item = []
    
    # get target images
    for f in os.listdir(foldername):
        if f.startswith(imgName) and f.endswith(".png"):
            lights.append(f[-6:-4])

            img = cv2.imread(os.path.join(relighted_imgPath, imgName, f))
            row, col, _ = img.shape
            img = cv2.resize(img, (512, 512))
            Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

            targetL = Lab[:,:,0]
            targetL = targetL.astype(np.float32)/255.0
            targetL = targetL.transpose((0,1))
            targetL = targetL[None,None,...]
            #targetL = Variable(torch.from_numpy(inputL).cuda())
            targetL = Variable(torch.from_numpy(targetL))
            all_targetL_for_item.append(targetL)
            all_imgname.append(imgName)

    # get input lights
    for l in lights:
        sh = np.loadtxt(os.path.join(foldername, imgName+"_light_"+l+'.txt'))
        sh = sh[0:9]
        sh = sh * 0.7
        sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
        #sh = Variable(torch.from_numpy(sh).cuda())
        sh = Variable(torch.from_numpy(sh))
        all_inputsh_for_item.append(sh)

    # the input image and target lighting are of the same data
    # the target image and input lighting are of the same data
    tmp = list(zip(all_targetL_for_item, all_inputsh_for_item))
    random.shuffle(tmp)
    all_inputL_for_item, all_targetsh_for_item = zip(*tmp)
    all_inputL_for_item = [a for a in all_inputL_for_item]
    all_targetsh_for_item = [a for a in all_targetsh_for_item]

    all_inputsh = all_inputsh + all_inputsh_for_item
    all_inputL = all_inputL + all_inputL_for_item
    all_targetL = all_targetL + all_targetL_for_item
    all_targetsh = all_targetsh + all_targetsh_for_item

optimizer = torch.optim.Adam(hourglass_network.parameters())

epochs = 14
for epoch in range(epochs):
    print("epoch =", epoch)
    last_img_name = ""
    features = []
    for i in range(len(all_targetL)):
        # skip training
        if epoch < 5:
            outputImg, outputSH  = hourglass_network(all_inputL[i], all_inputsh[i], 4)
        elif epoch >= 5 and epoch <= 7:
            outputImg, outputSH  = hourglass_network(all_inputL[i], all_inputsh[i], 8 - epoch)
        else:
            outputImg, outputSH  = hourglass_network(all_inputL[i], all_inputsh[i], 0)
        feature = hourglass_network.light.faceFeat

        # different loss functions for different epochs
        if epoch < 10 or i == 0:
            loss = image_loss(outputImg, outputSH, all_targetL[i], all_targetsh[i])
        else:
            if all_imgname[i] == last_img_name:
                loss = image_loss(outputImg, outputSH, all_targetL[i], all_targetsh[i]) + 0.5 * feature_loss(feature, features[-1])
                features.append(feature.detach())
            else:
                loss = image_loss(outputImg, outputSH, all_targetL[i], all_targetsh[i])
                features = [feature.detach()]
        
            last_img_name = all_imgname[i]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


outputImg = outputImg[0].cpu().data.numpy()
outputImg = outputImg.transpose((1,2,0))
outputImg = np.squeeze(outputImg)
outputImg = (outputImg*255.0).astype(np.uint8)
Lab[:,:,0] = outputImg
resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
resultLab = cv2.resize(resultLab, (col, row))

cv2.imshow('img',resultLab)
cv2.waitKey(0)


torch.save(hourglass_network.state_dict(),'my_trained_model')