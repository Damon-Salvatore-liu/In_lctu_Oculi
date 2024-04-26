import pickle, os, cv2
import numpy as np
import sys
import torch
from torch.utils.data import Dataset
from PIL import Image

sys.path.append('../')
import utils.img_utils.proc_img as ulib


class EyeDataset(Dataset):

    def __init__(self, anno_path, data_dir, transforms=None):
        self.anno_path = anno_path
        self.data_dir = data_dir
        self.transforms = transforms
        # Load annotations
        with open(self.anno_path, 'rb') as f:
            self.annos = pickle.load(f) # frames x 2
        
        self.eye_imgs = []
        self.blink_labels = []
        self._get_imgs()
        self.blink_labels = torch.tensor(self.blink_labels)

        
    def __len__(self):
        return len(self.annos)
    
    def __getitem__(self, index):
        img = self.eye_imgs[index]  
        label = self.blink_labels[index]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label
    
    def _get_imgs(self):
        for sample in self.annos:
            im_name, label = sample
            im_dir = self.data_dir + im_name + '.png'
            # im = Image.open(im_dir) 这样会有系统文件打开数量限制
            im_array = cv2.imread(im_dir)
            im = Image.fromarray(im_array)
            self.eye_imgs.append(im)
            self.blink_labels.append(label)            
    
    

    
if __name__ == "__main__":
    test = EyeDataset(
        anno_path='../datas/cnn/test.p',
        data_dir='../datas/cnn/',
    )
    print('完成')