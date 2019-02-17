#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from torch.utils.data import Dataset
import os
import glob
from pathlib import Path
from skimage.transform import radon
from skimage.util import random_noise
from torch.utils import data
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
class data_generator(Dataset):
    def __init__(self,root_dir,save_dir,n,n_proj,noise=True,transform=None):
        self.root_dir=root_dir
        self.size = sum(1 for f in root_dir if os.path.isfile(os.path.join(root_dir, f)) and f[0] != '.')
        self.n=n
        self.transform=transform
        self.save_dir=save_dir
        self.n_proj=n_proj
        self.noise=noise

    def __getitem__(self,original_file):
        '''
        '''
    def __len__(self):
        '''
        returns the number of output files
        '''
        return self.n*self.size
    
    def __call__(self):
        '''
        '''
        images=glob.glob(self.root_dir+"*.png")
        for image in images:
            img = io.imread(image,as_gray=True)
            fname=os.path.splitext(os.path.basename(image))[0]
            print("fname",fname)
            for i in range(self.n):
                img_new = self.transform(img)
                self.save(img_new,"{}_".format(fname)+"{}".format(i))
                sino=radon(img_new,theta=np.arange(180,step=180/self.n_proj),circle=True)
                #print(max(sino.flatten()),min(sino.flatten()))
                sino=(sino-min(sino.flatten()))/(max(sino.flatten())-min(sino.flatten()))
                if self.noise:
                    sino=random_noise(sino,'poisson')
                self.save(sino,"{}_".format(fname)+"{}_".format(i)+"sino")
        
    def save(self,data, name):
        '''
        '''
        io.imsave(self.save_dir+name+".png",data) 
    
        


# In[ ]:


from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        return img


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


# In[ ]:


#dataset_new=data_generator("C:/Users/ucsds/Downloads/Medical/test/","C:/Users/ucsds/Downloads/Medical/output/",5,200,noise=False,transform=RandomCrop(512))
#dataset_new()


# In[ ]:


print(Path("C:\Users\ucsds\Downloads\Medical\test"))


# In[ ]:


from torchvision import transforms,datasets
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
data_transform = transforms.Compose([
        transforms.RandomSizedCrop(512),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
dataset=datasets.ImageFolder("C:/Users/ucsds/Downloads/Medical/test/",transform=data_transform)
treain=data.DataLoader(dataset,batch_size=3,num_workers=2)


# In[ ]:


from scipy.io import loadmat
train=loadmat("C:/Users/ucsds/Downloads/train_elips.mat")


# In[5]:


import image_util
data_loader=image_util.ImageDataProvider_mat("C:/Users/ucsds/Downloads/train_elips.mat",is_flipping=False)
data_loader_test=image_util.ImageDataProvider_mat("C:/Users/ucsds/Downloads/test_elips.mat",shuffle_data=False,is_flipping=False)


# In[ ]:


import os 
print(os.getcwd())


# In[6]:


print(dir(data_loader))


# In[ ]:




