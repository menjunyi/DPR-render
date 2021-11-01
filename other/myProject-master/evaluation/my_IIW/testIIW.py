import sys
sys.path.append('model')
sys.path.append('utils')
from defineLoss import *
from loadData_IIW import *
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import os

baseFolder = '../data/IIW/'

# parameters are borrowed from the paper
criterion_test = WhdrTestLoss_Paper(delta=0.1)

transformer = transforms.Compose([resizeImg(224), ToTensor()])

testLoaderHelper = IIWData(dataFolder=os.path.join(baseFolder, 'data'),
				labelFolder=os.path.join(baseFolder, 'label_forPytorch'),
				fileListName=os.path.join(baseFolder, 'testName.list'),
				transform = transformer)

testLoader = torch.utils.data.DataLoader(testLoaderHelper, 
				batch_size=5, shuffle=False, num_workers=4)

def getImage(image, output_albedo, output_shading, count, resultPath):
		'''
			compute the loss for the mini-batch
		'''
		if not os.path.exists(resultPath):
				os.makedirs(resultPath)

		for i in range(output_albedo.size()[0]):
				imgAlbedo = np.squeeze(output_albedo[i].cpu().data.numpy())
				imgShading = np.squeeze(output_shading[i].cpu().data.numpy())

				imgAlbedo = imgAlbedo.transpose((1,2,0))
				imgShading = imgShading.transpose((1,2,0))
				imgAlbedo = ((imgAlbedo-np.min(imgAlbedo))/(np.max(imgAlbedo) - np.min(imgAlbedo))*255).astype(np.uint8)
				imgShading = ((imgShading-np.min(imgShading))/(np.max(imgShading) - np.min(imgShading))*255).astype(np.uint8)
				io.imsave(os.path.join(resultPath, 'shading_{:04d}_{:02d}.png'.format(count, i)), imgShading)
				io.imsave(os.path.join(resultPath, 'albedo_{:04d}_{:02d}.png'.format(count, i)), imgAlbedo)


		return output_albedo

def main(savePath, modelPath):
		my_network = torch.load(os.path.join(modelPath, 'trained_model.t7'))
		my_network.cuda()
		my_network.train(False)
		test_loss = 0
		count = 0
		for i, data in enumerate(testLoader, 0):
				inputs, labels, numComparisons = data
				inputs, labels, numComparisons = Variable(inputs.cuda(),volatile=True).float(), \
								Variable(labels, volatile=True), \
								Variable(numComparisons, volatile=True)
				output_albedo, output_shading = my_network(inputs)
				getImage(inputs, output_albedo, output_shading, count, savePath)
				outputs_albedo = torch.mean(output_albedo, dim=1, keepdim=True)
				loss = criterion_test(outputs_albedo, labels, numComparisons)
				test_loss += loss.data[0]
				count = count + 1
		print test_loss
		print count
		print 'test loss: %.3f' % (test_loss/count)
		
		print('Finished testing')

if __name__ == '__main__':
		savePath = sys.argv[1]
		modelPath = sys.argv[2]
		if not os.path.exists(savePath):
				os.makedirs(savePath)
		main(savePath, modelPath)
