# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.

'''
author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

#import cv2
import glob
import numpy as np
from PIL import Image
import h5py
import scipy.io as sio

import matplotlib.pyplot as plt
from skimage.io import imread
#from skimage import data_dir
from skimage.transform import radon,iradon, rescale
import pdb 
DEBUG = False
def flipping(img,gt):
    if np.random.rand(1)>0.5:
        out=np.fliplr(img)
        out_gt=np.fliplr(gt)
    else:
        out=img
        out_gt=gt
    if np.random.rand(1)>0.5:
        out=np.flipud(out)
        out_gt=np.flipud(out_gt)
    return out, out_gt
def computeRegressedSNR(rec,oracle):
    '''
        HDK: This SNR is not matching the SNR that was create for noise addition.
    '''
    # type convert oracle and recreated data
    oracle = np.array(oracle,dtype=float)
    #print("oracle = ",oracle)
    rec = np.array(rec,dtype=float)
    #print("rec = ",rec)
    
    # perform required operations
    sumP = oracle.sum()
    sumI = rec.sum()
    IP = oracle*rec
    sumIP = IP.sum()
    I2 = rec**2
    sumI2 = I2.sum()
    (Nrows_oracle,Ncols_oracle) = np.shape(oracle)
    Noracle = Nrows_oracle*Ncols_oracle
    A = np.array(([sumI2,sumI],[sumI,Noracle]),dtype=float)
    b = np.array(([sumIP],[sumP]),dtype=float)
    A_pinv = np.linalg.pinv(A)
    c = np.matmul(A_pinv,b)
    rec = c[0]*rec + c[1]
    diff2 = (oracle - rec)**2
    err = diff2.sum()
    P2 = oracle**2
    SNR = 10*np.log10(P2.sum()/err)
    
    return SNR

def iRadon(Sinogram, snr_db, DownSampRatio, DEBUG = False):
    # iRadon transform

    Sinogram = Sinogram.transpose()
    Nviews  = Sinogram.shape[0]
    NumMeas = Sinogram.shape[1]
    NumImage = Sinogram.shape[-1]
    FBPimages = np.zeros((512,512,1,NumImage))
    theta = np.linspace(180.0/NumMeas,180.0,NumMeas/DownSampRatio,endpoint=False)
    
    # This loop can be parallized for speedup, seems very slow right now
    for i in range(NumImage):
        if 0:
            # add Gaussian Noise to input sinogram image
            norm_sinogram = np.linalg.norm(Sinogram[:,:,0,i], 'fro')
            sigma_noise = 1 / np.sqrt(10 ** (snr_db / 10) / (norm_sinogram ** 2 / (Nviews * NumMeas)))
            noise = np.random.normal(0, sigma_noise, (Nviews, NumMeas))
            # print("Noise = ",noise)
            NoisySinogram = Sinogram[:,:,0,i] + noise
            #FBPimages[:, :, 0, i] = iradon(NoisySinogram[:,np.arange(0, NumMeas, DownSampRatio), 0, i], theta, output_size=512, circle=True)
            FBP = iradon(NoisySinogram[:, np.arange(0, NumMeas, DownSampRatio)], theta, output_size=512, circle=False)
            FBPimages[:, :, 0, i] = FBP 
        else:
            FBP = iradon(Sinogram[:, np.arange(0, NumMeas, DownSampRatio),0,i], theta, output_size=514, circle=False)
            FBP = FBP[2:,2:]
            NormFbp = np.linalg.norm(FBP, 'fro')
            sigma_noise = 1 / np.sqrt(10 ** (snr_db / 10) / (NormFbp ** 2 / (512 * 512)))
            noise = np.random.normal(0, sigma_noise, (512, 512))
            
            FBPimages[:, :, 0, i] = FBP  + noise

        if(DEBUG):
            plt.imsave('./testimg/fbpimage.png',FBPimages[:, :, 0, 0])
            #plt.figsave('./testimg/fbpimage.png')
            pdb.set_trace()
            break;
    return FBPimages

class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """

    channels = 1
    n_class = 1


    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        data, label = self._next_data()

        norm_fac=1.#500.
        train_data = data/norm_fac #self._process_data(data)
        labels = label/norm_fac #self._process_labels(label)

        train_data, labels = self._post_process(train_data, labels)

        nx = train_data.shape[1]
        ny = train_data.shape[0]

        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class),

    def _process_labels(self, label):
        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = ~label
            return labels

        return label

    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data

    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation

        :param data: the data array
        :param labels: the label array
        """
        return data, labels

    def __call__(self, n):
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]

        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))

        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels

        return X, Y

class SimpleDataProvider(BaseDataProvider):
    """
    A simple data provider for numpy arrays.
    Assumes that the data and label are numpy array with the dimensions
    data `[n, X, Y, channels]`, label `[n, X, Y, classes]`. Where
    `n` is the number of images, `X`, `Y` the size of the image.

    :param data: data numpy array. Shape=[n, X, Y, channels]
    :param label: label numpy array. Shape=[n, X, Y, classes]
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2

    """

    def __init__(self, data, label, a_min=None, a_max=None, channels=1, n_class = 2):
        super(SimpleDataProvider, self).__init__(a_min, a_max)
        self.data = data
        self.label = label
        self.file_count = data.shape[0]
        self.n_class = n_class
        self.channels = channels

    def _next_data(self):
        idx = np.random.choice(self.file_count)
        return self.data[idx], self.label[idx]


class ImageDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")

    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2

    """

    def __init__(self, search_path, a_min=None, a_max=None, data_suffix=".tif", mask_suffix='_mask.tif', shuffle_data=True, n_class = 2):
        super(ImageDataProvider, self).__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class

        self.data_files = self._find_data_files(search_path)

        if self.shuffle_data:
            np.random.shuffle(self.data_files)

        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))

        img = self._load_file(self.data_files[0])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]

    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return [name for name in all_files if self.data_suffix in name and not self.mask_suffix in name]


    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path), dtype)
        # return np.squeeze(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0
            if self.shuffle_data:
                np.random.shuffle(self.data_files)

    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = image_name.replace(self.data_suffix, self.mask_suffix)

        img = self._load_file(image_name, np.float32)
        label = self._load_file(label_name, np.bool)

        return img,label



class ImageDataProvider_hdf5(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")

    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2

    """

    def __init__(self, search_path, SinoVar, GrdTruthVar, SnrDb, DownSampRatio, a_min=None, a_max=None, shuffle_data=True, is_flipping=True, n_class= 1, DEBUG= False):
        super(ImageDataProvider_hdf5, self).__init__(a_min, a_max)
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.is_flipping= is_flipping
        self.n_class = n_class

        self.data_files = self._find_data_files(search_path)
        Sinogram=self._load_file(self.data_files[0],SinoVar)
        print(np.shape(Sinogram))
        self.data_train = iRadon(Sinogram, snr_db= SnrDb, DownSampRatio= DownSampRatio, DEBUG= DEBUG)
        
        #self.data_train=self._load_file(self.data_files[0],'sparse')
        #self.data_label=self._load_file(self.data_files[0],'label')
        self.data_label=self._load_file(self.data_files[0],GrdTruthVar)
        self.data_label = self.data_label.transpose()
        print(np.shape(self.data_label))
        if DEBUG:
            #plt.imshow(self.data_train[:,:,0,0])
            #plt.show()
            #plt.imshow(self.data_label[:,:,0,0])
            #plt.show()
            snr = computeRegressedSNR(self.data_train[:,:,0,0],
                                      self.data_label[:,:,0,0])
            print("SNR input %0.4f applied %0.4f"%(SnrDb, snr))
            temp = self.data_train[:,:,0,0] - self.data_label[:,:,0,0]
            plt.imsave('./testimg/LabelDiffFbp.png',temp)
            pdb.set_trace()

        self.tN=self.data_train.shape[-1]
        self.ids=np.arange(self.tN)
        if self.shuffle_data:
            #np.random.shuffle(self.data_files)
            np.random.shuffle(self.ids)

        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))

        #img = self._load_file(self.data_files[0])
        #self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        self.channels=n_class

    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return all_files

    def _load_file(self, path, opt,dtype=np.float32):
        f_handle=h5py.File(path,"r")
        data=np.array(f_handle[opt])
        return data
        #return np.array(Image.open(path), dtype)
        # return np.squeeze(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= self.tN:#len(self.data_files):
            self.file_idx = -1
            if self.shuffle_data:
                np.random.shuffle(self.ids)
                #np.random.shuffle(self.data_files)

    def _next_data(self):
        self._cylce_file()
        id=self.ids[self.file_idx]
        #image_name = self.data_files[self.file_idx]
        #label_name = image_name.replace(self.data_suffix, self.mask_suffix)

        img = self.data_train[:,:,id]#self._load_file(image_name, np.float32)
        label = self.data_label[:,:,id]#self._load_file(label_name, np.bool)
        if self.is_flipping:
            img,label=flipping(img,label)

        return img,label

    def __getitem__(self,item):
        self._cylce_file()
        id=self.ids[self.file_idx]
        #image_name = self.data_files[self.file_idx]
        #label_name = image_name.replace(self.data_suffix, self.mask_suffix)

        img = self.data_train[:,:,0,item]#self._load_file(image_name, np.float32)
        label = self.data_label[:,:,0,item]#self._load_file(label_name, np.bool)
        if self.is_flipping:
            img,label=flipping(img,label)

        return np.expand_dims(img,axis=0),np.expand_dims(label,axis=0)

    
    def __len__(self):
        return self.data_train.shape[-1]




class ImageDataProvider_mat(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")

    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2

    """

    def __init__(self, search_path, a_min=None, a_max=None, shuffle_data=True, is_flipping=True,n_class = 1):
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.is_flipping= is_flipping
        self.n_class = n_class

        self.data_files = self._find_data_files(search_path)

        self.data_train=self._load_file(self.data_files[0],'sparse')
        self.data_label=self._load_file(self.data_files[0],'label')
        self.tN=self.data_train.shape[-1]
        self.ids=np.arange(self.tN)
        if self.shuffle_data:
            #np.random.shuffle(self.data_files)
            np.random.shuffle(self.ids)

        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))

        #img = self._load_file(self.data_files[0])
        #self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        self.channels=n_class

    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return all_files

    def _load_file(self, path, opt,dtype=np.float32):
        mat_contents=sio.loadmat(path)
        data=np.squeeze(mat_contents[opt])
        return data

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= self.tN:#len(self.data_files):
            self.file_idx = -1
            if self.shuffle_data: 
                np.random.shuffle(self.ids)
                #np.random.shuffle(self.data_files)

    def __getitem__(self,item):
        self._cylce_file()
        id=self.ids[self.file_idx]
        #image_name = self.data_files[self.file_idx]
        #label_name = image_name.replace(self.data_suffix, self.mask_suffix)

        img = self.data_train[:,:,item]#self._load_file(image_name, np.float32)
        label = self.data_label[:,:,item]#self._load_file(label_name, np.bool)
        if self.is_flipping:
            img,label=flipping(img,label)

        return np.expand_dims(img,axis=0),np.expand_dims(label,axis=0)

    def __len__(self):
        return self.data_train.shape[-1]

if __name__ == '__main__':
    DataPath = "../CT_Data/CT_Head_Neck_Validate.mat" # ".\Data\CT_Head_Neck_Validate.mat"
    ImageDataProvider_hdf5( DataPath,
                            SinoVar='Sinogram',
                            GrdTruthVar='FBPImage',
                            SnrDb = 40,
                            DownSampRatio=1,
                            is_flipping=False,
                            DEBUG = True)
