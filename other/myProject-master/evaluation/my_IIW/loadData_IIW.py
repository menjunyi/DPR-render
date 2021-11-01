import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import numpy as np
import cv2

class IIWData(Dataset):
		'''
			loading IIW data sets
		'''

		def __init__(self, dataFolder, labelFolder, fileListName, transform=None):
				'''
					dataFolder: contains images
					labelFolder: contains labels
					fileListName: all file names
				'''

				self.fileList = []
				with open(fileListName) as f:
						for line in f:
								self.fileList.append(line.strip())

				self.dataFolder = dataFolder
				self.labelFolder = labelFolder
				self.transform = transform
				self.maxNum = 1181   # empirically, the maximum number of 
									 # comparisons is 1181
		def __len__(self):
				return len(self.fileList)

		def __getitem__(self, idx):
				fileName = self.fileList[idx]
				imgName = os.path.join(self.dataFolder, fileName + '.png')
				image = io.imread(imgName)
				labelName = os.path.join(self.labelFolder, fileName + '.csv')
				labelCSV = pd.read_csv(labelName)
				label = np.zeros((self.maxNum, 6))
				numComparisons = np.zeros((1))
				numComparisons[0] = labelCSV.shape[0]
				for i in range(labelCSV.shape[0]):
						item = labelCSV.ix[i]
						tmpLabel = [float(x) for x in item]
						tmpLabel = tmpLabel[2:]
						label[i] = tmpLabel
						#label.append(tmpLabel)
				#label = np.array(label)
				if self.transform:
						image, label, numComparisons = self.transform([image, label, numComparisons])
				return image, label, numComparisons
	

class resizeImg(object):
		def __init__(self, output_size):
				self.size = output_size
		def __call__(self, sample):
				img, label, numComparisons = sample
				label[:,0:4] = (label[:,0:4]*self.size).astype(int)
				img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
				img = 1.0*img/255.0
				return img, label, numComparisons

class ToTensor(object):
		"""Convert ndarrays in sample to Tensors."""
		def __call__(self, sample):
				image, label, numComparisons = sample
				# swap color axis because
				# numpy image: H x W x C
				# torch image: C X H X W
				image = image.transpose((2, 0, 1))
				return torch.from_numpy(image), torch.from_numpy(label), torch.from_numpy(numComparisons)
