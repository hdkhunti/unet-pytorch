import argparse
import os
import random
import shutil
import time
import warnings
import sys
import cv2
import numpy as np
import pdb
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from image_util import ImageDataProvider_mat
from image_util import ImageDataProvider_hdf5
from image_util import computeRegressedSNR
from model import UNet

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--TrainData',type=str,
                    help='Training .mat file path')
parser.add_argument('--TestData',type=str,
                    help='Test .mat file path')
parser.add_argument('--DownSample', type=str,
                    help='Python array of Down Sample ratio [10 20 30]')
parser.add_argument('--SnrDb', type=str,
                    help='Python array of SnrDb [10  20  30]')
parser.add_argument('--OutDir',type=str,
                    help ='Output directory path it should exist')

parser.add_argument('--TrainedModelPath',type=str,
                    help ='Trained Model Path/filename ')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # create model
    model = UNet(1, depth=5, merge_mode='concat')
    model = model.cuda(args.gpu)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.L1Loss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    print(train_sampler)
    # Get SNR and Down Sample Values
    SnrDb = eval(args.SnrDb)
    DownSamp = eval(args.DownSample)
    print("Down Sample value:",DownSamp)
    print("Snr Values:", SnrDb)
    print("Training Data: %s" % (args.TrainData))
    print("Test Data: %s"%args.TestData)
    # add forward slash
    if (args.OutDir[-1] != '/'):
        args.OutDir += '/'
    # Train data
    

    TrainDataPath = args.TrainData
    loss_train = np.zeros((args.epochs,len(DownSamp),len(SnrDb)), dtype=float)
    loss_test = np.zeros((args.epochs,len(DownSamp),len(SnrDb)), dtype=float)
    RecSnrMean = np.zeros((args.epochs,len(DownSamp),len(SnrDb)), dtype=float)
    x=np.linspace(0,args.epochs,args.epochs)
    k=1
    for SnrIdx in range(len(SnrDb)):
        ## updating the model for each SNR IDX
        model = UNet(1, depth=5, merge_mode='concat')
        model = model.cuda(args.gpu)
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        
        ######
        for DownSampIdx in range(len(DownSamp)):
            loss_data = Loss_Data(args.epochs)
            aStr ="Snr %d DownSamp %d" % (SnrDb[SnrIdx], DownSamp[DownSampIdx]) 
            print(aStr)
            dirName = aStr.replace(' ','_')

            if not os.path.exists(args.OutDir+dirName):
                os.makedirs(args.OutDir+dirName)
                print("Directory " , dirName ,  " Created ")
            else:    
                print("Directory " , dirName ,  " already exists")    
            
            
            #TestDataPath = ".\Data\CT_Head_Neck_Test.mat"
            
            TestDataPath = args.TestData
            val_loader = torch.utils.data.DataLoader(\
                ImageDataProvider_hdf5(TestDataPath,
                                        SinoVar='Sinogram',
                                        GrdTruthVar='FBPImage',
                                        DownSampRatio=DownSamp[DownSampIdx],
                                        SnrDb=SnrDb[SnrIdx],
                                        is_flipping=False),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

            if args.evaluate:
                model.load_state_dict(torch.load(args.TrainedModelPath))
                model.eval()
                loss_test[0, DownSampIdx, SnrIdx], RecSnrMean[0,DownSampIdx,SnrIdx] = \
                                    validate(val_loader, model, criterion, 0, args, ImageDir=dirName)
                    
                #validate(val_loader, model, criterion, args,ImageDir=dirName)
                #not return, continue from the start of the loop
                continue 

            
            train_loader = torch.utils.data.DataLoader(
                ImageDataProvider_hdf5( TrainDataPath,
                                        SinoVar='Sinogram',
                                        GrdTruthVar='FBPImage',
                                        DownSampRatio=DownSamp[DownSampIdx],
                                        SnrDb=SnrDb[SnrIdx],
                                        is_flipping=False),
                batch_size=args.batch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler)
            
            for epoch in range(args.start_epoch, args.epochs):
                if args.distributed:
                    train_sampler.set_epoch(epoch)
                adjust_learning_rate(optimizer, epoch, args)

                # train for one epoch
                loss_train[epoch,DownSampIdx,SnrIdx] = train(train_loader, model, criterion, optimizer, epoch, args)

                # evaluate on validation set
                loss_test[epoch,DownSampIdx,SnrIdx], RecSnrMean[epoch,DownSampIdx,SnrIdx] = \
                                    validate(val_loader, model, criterion, epoch, args, ImageDir=dirName)
                
            loss_data.update(SnrDb[SnrIdx],
                            DownSamp[DownSampIdx],
                            loss_train[:,DownSampIdx,SnrIdx],
                            loss_test[:,DownSampIdx,SnrIdx])

            file_name =args.OutDir+str(SnrDb[SnrIdx])+'_'+str(DownSamp[DownSampIdx])+'.pkl'
            #print('SNR %f DS %F', (loss_data.SNRdb,loss_data.Downsampling))
            file_loss_data = open(file_name,'w')
            pickle.dump(loss_data,file_loss_data)
            file_loss_data.close()
                
            ##To check
            #pkl_file = open(file_name, 'r')
            #data1 = pickle.load(pkl_file)
            #print("Test Print")
            #print(data1.SNRdb,data1.Downsampl)
            ####
            plt.figure(1)
            plt.plot(x,loss_train[:,DownSampIdx,SnrIdx],label="DownSampIdx = %0.2f Snr = %0.2f dB "%((DownSamp[DownSampIdx]),                     SnrDb[SnrIdx]))
            plt.ylabel('L1_Loss')
            plt.xlabel('Epochs')
           
            plt.figure(2)
            plt.plot(x,RecSnrMean[:,DownSampIdx,SnrIdx],label="DownSampIdx = %0.2f Snr = %0.2f dB "%((DownSamp[DownSampIdx]),                 SnrDb[SnrIdx]))
            plt.ylabel('Reconstructed Image SNR')
            plt.xlabel('Epochs')
            
            ## Save model 
            file_name_model = args.OutDir+aStr.replace(' ','_')+'_UnetModel.ph'
            torch.save(model.state_dict(), file_name_model)
            
    
    plt.figure(1)
    plt.legend()
    plt.savefig(args.OutDir+'LossVsEpochs.png')
    plt.figure(2)
    plt.legend()
    plt.savefig(args.OutDir+'RecSnrVsEpochs.png')
    plt.clf()
    # Save the matrix 
    file_name =args.OutDir+'LossVsSnrVsDwnsamp'+'.pkl'
    with open(file_name, 'w') as f:
        pickle.dump([SnrDb, DownSamp, loss_train, loss_test, RecSnrMean],f)
    

    


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        #plt.imshow(input[0,:,:,:].squeeze().data.numpy(),cmap='gray')
        #plt.show()
        
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)#async=True) #non_blocking=True)#
        target = target.cuda(args.gpu, non_blocking=True)#async=True) #non_blocking=True)

        # compute output
        output = model(input.type(torch.cuda.FloatTensor))
        loss = criterion(output, target.type(torch.cuda.FloatTensor))

        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
    return losses.avg


def validate(val_loader, model, criterion, epoch, args, ImageDir='/' ):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    #Reconstructed Image SNR
    RecSnr = np.zeros((len(val_loader), 1), dtype=float)

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu,non_blocking=True)#async=True)# non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)#async=True)#non_blocking=True)

            # compute output
            output = model(input.type(torch.cuda.FloatTensor))

            #write out the sample
            if i == 0:
                imgname = args.OutDir + ImageDir + '/' + str(epoch) + '.jpg'
                plt.imsave(imgname, output[0,:,:,:].squeeze().data.cpu().numpy())


            loss = criterion(output, target.type(torch.cuda.FloatTensor))

            losses.update(loss.item(), input.size(0))
            # Measure SNR 
            #pdb.set_trace()
            output = output.cpu()
            target = target.cpu()
            RecSnr[i] = computeRegressedSNR(output[0,0,:,:],target[0,0,:,:])
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses))
    return losses.avg, RecSnr.mean()

class Loss_Data(object):
    def __init__(self,epochs):
        self.reset(epochs)
        
    def reset(self,epochs):
        self.SNRdb = 0
        self.Downsampling = 0
        self.Train_Loss = np.zeros((epochs,1),dtype=float)
        self.Validate_Loss = np.zeros((epochs,1),dtype = float)
        
    def update(self,SNRdb, Downsampling, Train_Loss,Validate_Loss):
        self.SNRdb = SNRdb
        self.Downsampling = Downsampling
        self.Train_Loss = Train_Loss
        self.Validate_Loss = Validate_Loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
