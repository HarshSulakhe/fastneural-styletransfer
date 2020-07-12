import torch as th
import os
from PIL import Image

class COCODataset(th.utils.data.Dataset):

    def __init__(self,root,transform):
        self.root = root
        self.transform  = transform
        self.imglist = os.listdir(self.root)

    def __getitem__(self,index):
        image = Image.open(self.root+self.imglist[index]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return (len(self.imglist))
