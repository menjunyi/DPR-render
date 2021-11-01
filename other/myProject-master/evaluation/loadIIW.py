import cv2
import torch.utils.data as data
import pickle
import numpy as np
import torch
from skimage import io

def make_dataset(list_dir):
    file_name = list_dir + "img_batch.p"
    #file_name = list_dir + "SAW_train_batch.p"
    images_list = pickle.load( open( file_name, "rb" ) )
    return images_list

class IIW_ImageFolder():
    '''
        image loader for IIW dataset, adpated from code:
            Learning Intrinsic Image Decomposition from Watching the World
        Z. Li and N. Snavely, CVPR 2018
     '''

    def __init__(self, root, list_dir, imageSize, mode, is_flip, transform=None, 
                 loader=None):
        # load image list from hdf5
        img_list = make_dataset(list_dir)
        if len(img_list) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.list_dir = list_dir

        self.img_list = img_list
        self.transform = transform
        self.loader = loader
        self.current_o_idx = mode
        self.set_o_idx(mode)
        x = np.arange(-1, 2)
        y = np.arange(-1, 2)
        self.height = imageSize
        self.width = imageSize
        self.X, self.Y = np.meshgrid(x, y)

    def set_o_idx(self, o_idx):
        self.current_o_idx = o_idx

    def iiw_loader(self, img_path):
        
        img_path = img_path[-1][:-3]
        img_path = self.root + img_path
        img = np.float32(io.imread(img_path))/ 255.0
        original_shape = img.shape

        # NOTE: we use cv2 to resize image
        img = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_CUBIC)

        return img, original_shape


    def __getitem__(self, index):
        targets_1 = {}

        # IIW
        img_id = self.img_list[self.current_o_idx][index].split('/')[-1][0:-6]
        judgement_path = self.root + img_id + 'json'

        img, oringinal_shape = self.iiw_loader(self.img_list[self.current_o_idx][index].split('/'))

        targets_1['path'] = self.img_list[self.current_o_idx][index]
        targets_1["judgements_path"] = judgement_path
        targets_1["oringinal_shape"] = oringinal_shape

        final_img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2,0,1)))).contiguous().float()
        return final_img, targets_1, img_id
        ## saw
        #img_id = self.img_list[self.current_o_idx][index]
        #return [], [], img_id



    def __len__(self):
        return len(self.img_list[self.current_o_idx])

class IIWTESTDataLoader():
    def __init__(self,_root, _list_dir, imageSize, mode):

        transform = None
        # transform = transforms.Compose(transformations)

        # Dataset A
        # dataset = ImageFolder(root='/phoenix/S6/zl548/AMOS/test/', \
                # list_dir = '/phoenix/S6/zl548/AMOS/test/list/',transform=transform)
        # testset 
        dataset = IIW_ImageFolder(root=_root, \
                list_dir =_list_dir, imageSize=imageSize, mode= mode, is_flip = False, transform=transform)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size= 16, shuffle= False, num_workers=int(1))
        self.dataset = dataset
        self.iiw_data = data_loader

    def name(self):
        return 'IIWTESTDataLoader'

    def load_data(self):
        return self.iiw_data

    def __len__(self):
        return len(self.dataset)



