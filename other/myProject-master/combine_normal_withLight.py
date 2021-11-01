import numpy as np
import os
import cv2
import sys

sourceDirIIW = '../data/IIW/data/'
IIW_testList = '../data/IIW/testName.list'

sourceDir_SUNCG= '/scratch1/intrinsicImage/synthetic_SUNCG/'
sourceDir_albedo = os.path.join(sourceDir_SUNCG, 'albedo')
sourceDir_shading = os.path.join(sourceDir_SUNCG, 'shading')
sourceDir_img = os.path.join(sourceDir_SUNCG, 'images_color')
sourceDir_normal = '/scratch1/data/SUNCG_groundTruth/SUNCG_normal/'
SUNCG_list = os.path.join(sourceDir_SUNCG, 'testing.list')

def generateImage_SUNCG(result_SUNCG, savePath):
    img_size = 128
    count = 0
    with open(SUNCG_list) as f:
        for line in f:
            if count > 250:
                break
            line = line.strip()
            img = cv2.imread(os.path.join(sourceDir_img, line))
            albedo = cv2.imread(os.path.join(sourceDir_albedo, line))
            shading = cv2.imread(os.path.join(sourceDir_shading, line))
            normal = cv2.imread(os.path.join(sourceDir_normal, line[0:-8] + '_norm_camera.png'))
            
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            albedo = cv2.resize(albedo, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            shading = cv2.resize(shading, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            normal = cv2.resize(normal, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            
            tmp_1 = count/20
            tmp_2 = count%20
            pred_albedo = cv2.imread(os.path.join(result_SUNCG, 'albedo_{:04d}_{:02d}.png'.format(tmp_1, tmp_2)))
            pred_shading = cv2.imread(os.path.join(result_SUNCG, 'shading_{:04d}_{:02d}.png'.format(tmp_1, tmp_2)))
            pred_normal = cv2.imread(os.path.join(result_SUNCG, 'normal_{:04d}_{:02d}.png'.format(tmp_1, tmp_2)))
            pred_lighting = cv2.imread(os.path.join(result_SUNCG, 'sphere_{:04d}_{:02d}.png'.format(tmp_1, tmp_2)))
            diff_normal = cv2.imread(os.path.join(result_SUNCG, 'diff_normal_{:04d}_{:02d}.png'.format(tmp_1, tmp_2)))
            
            pred_albedo = cv2.resize(pred_albedo, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            pred_shading = cv2.resize(pred_shading, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            pred_normal = cv2.resize(pred_normal, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            pred_lighting = cv2.resize(pred_lighting, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            diff_normal = cv2.resize(diff_normal, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            
            
            H, W, C = img.shape
            tmpImg = np.zeros((2*H, 5*W, C))
            tmpImg[0:H, 0:W, :] = img
            tmpImg[0:H, W:2*W, :] = albedo
            tmpImg[0:H, 2*W:3*W, :] = normal
            tmpImg[0:H, 3*W:4*W, :] = shading
            
            tmpImg[H:2*H, 0:W, :] = pred_lighting
            tmpImg[H:2*H, W:2*W, :] = pred_albedo
            tmpImg[H:2*H, 2*W:3*W, :] = pred_normal
            tmpImg[H:2*H, 3*W:4*W, :] = pred_shading
            tmpImg[H:2*H, 4*W:5*W :] = diff_normal
            
            cv2.imwrite(os.path.join(savePath, 'img_{:04d}.png'.format(count)), tmpImg.astype(np.uint8))
            print os.path.join(savePath, 'img_{:04d}.png'.format(count))
            count = count + 1

def generateImage_IIW(result_IIW,savePath):
		count = 0
		with open(IIW_testList) as f:
				for line in f:
						tmp_1 = count/10
						tmp_2 = count%10

						line = line.strip()
						img = cv2.imread(os.path.join(sourceDirIIW, line + '.png'))
						img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
						pred_albedo = cv2.imread(os.path.join(result_IIW, 'albedo_{:04d}_{:02d}.png'.format(tmp_1, tmp_2)))
						pred_shading = cv2.imread(os.path.join(result_IIW, 'shading_{:04d}_{:02d}.png'.format(tmp_1, tmp_2)))

						tmpImg = np.zeros((256, 3*256, 3))
						tmpImg[:, 0:256, :] = img
						tmpImg[:, 256:2*256, :] = pred_albedo
						tmpImg[:, 2*256:3*256 :] = pred_shading
						cv2.imwrite(os.path.join(savePath, 'img_{:04d}.png'.format(count)), tmpImg.astype(np.uint8))
						count = count + 1
if __name__ == '__main__':
		resultPath = sys.argv[1]
		savePath = sys.argv[2]
		my_type = int(sys.argv[3])

		if not os.path.exists(savePath):
				os.makedirs(savePath)
		if my_type==0:
				generateImage_SUNCG(resultPath, savePath)
		else:
				generateImage_IIW(resultPath, savePath)

