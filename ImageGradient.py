
# coding: utf-8

# In[1]:


import torch 
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pdb

def fn_LoadImage(path, dtype=np.float32):
    return np.array(Image.open(path).convert("L"), dtype)

def ImageGrad(InputImage,device = 0):
    #InputImage = torch.from_numpy(InputImage)
    
    #pdb.set_trace()
    #InputImage = InputImage.view((1,1,InputImage.size()[0],InputImage.size()[1]))
    #InputImage = InputImage.cuda(device)
    #print(InputImage.size())

    InChannel = 1#InputImage.size()[0] # Gray scale input
    MiniBatch = 1#InputImage.size()[1] # This loss function should be of the same size of minibatch as the output

    Dx = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0,-1]])
    Dx = Dx.cuda(device)
    Dy = torch.t(Dx)
    Dx = Dx.view((InChannel, MiniBatch, 3, 3))
    Dy = Dy.view((InChannel, MiniBatch, 3, 3))
    #print(Dx) 
    #print(Dy)
    #print(Dx)

    #print(Dx.size())
    # Functional or nn module ? 
    ImDx = F.conv2d(InputImage, Dx,bias=None, stride=1, padding=0, dilation=1, groups=1)
    ImDy = F.conv2d(InputImage, Dy,bias=None, stride=1, padding=0, dilation=1, groups=1)

    return ImDx, ImDy

def GradLoss(RecIm, TargetIm):
    RecDx, RecDy = ImageGrad(RecIm)
    TarDx, TarDy = ImageGrad(TargetIm)
    #loss = 0.2*F.l1_loss(RecDx,TarDx) + 0.2*F.l1_loss(RecDy,TarDy) + 0.6*F.mse_loss(RecIm, TargetIm)
    loss = 0.25*F.mse_loss(RecDx,TarDx) + 0.25*F.mse_loss(RecDy,TarDy) + 0.5*F.l1_loss(RecIm, TargetIm)
    #loss = 0.25*F.smooth_l1_loss(RecDx,TarDx) + 0.25*F.smooth_l1_loss(RecDy,TarDy) + 0.5*F.smooth_l1_loss(RecIm, TargetIm)
    return loss

if __name__ == '__main__':
    device = 0
    # Sobel Filter 
    Image1 = './unet-pytorch2/TestImage_Run010319/Snr_1000_DownSamp_20/0.jpg'
    Image2 = './unet-pytorch2/TestImage_Run010319/Snr_1000_DownSamp_5/49.jpg'
    RecIm = fn_LoadImage(Image1)
    TarIm = fn_LoadImage(Image2)
    loss = GradLoss(RecIm,TarIm)
    #loss.backward()
    print(type(loss))
    # 
    if 0:
        InputImage = fn_LoadImage(Image1)
        ImDx, ImDy = ImageGrad(InputImage)


        plt.figure(1)
        ImDx = ImDx.cpu().numpy()

        plt.imsave('./Dx_0.png',ImDx[0,0,:,:])

        plt.figure(2)
        ImDy = ImDy.cpu().numpy()
        plt.imsave('./Dy_0.png',ImDy[0,0,:,:])


        InputImage = fn_LoadImage(Image2)
        ImDx2, ImDy2 = ImageGrad(InputImage)


        plt.figure(1)
        ImDx2 = ImDx2.cpu().numpy()
        #ImDx2 = ImDx2.numpy()
        plt.imsave('./Dx_49.png',ImDx2[0,0,:,:])

        plt.figure(2)
        ImDy2 = ImDy2.cpu().numpy()
        #ImDy2 = ImDy2.numpy()
        plt.imsave('./Dy_49.png',ImDy2[0,0,:,:])

        plt.imsave('./Dx_diff.png',ImDx.squeeze() - ImDx2.squeeze())
        plt.imsave('./Dy_diff.png',ImDy.squeeze() - ImDy2.squeeze())


