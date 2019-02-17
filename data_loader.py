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



import image_util
def dataloader(root_path,is_flipping= False, shuffle_data=False):
    data_loader=image_util.ImageDataProvider_mat("C:/Users/ucsds/Downloads/train_elips.mat",is_flipping=False)
    data_loader_test=image_util.ImageDataProvider_mat("C:/Users/ucsds/Downloads/test_elips.mat",shuffle_data=False,is_flipping=False)
    return {'data_loader':data_loader,'data_loader_test':data_loader_test}





