import torch
from image_util import ImageDataProvider_mat
from model import UNet

import scipy.io as sio

import os
import sys
import copy
import torch.multiprocessing

import string
import random
import argparse
import time
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

from tqdm import tqdm
import math

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

parser = argparse.ArgumentParser()
parser.add_argument('--n_worker', default=0, type=int,
                    help='number of data loading workers (default: 4)')

parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--run_id', default='', type=str, metavar='run_id')
parser.add_argument('--dump_dir', default='./logs/train_feature', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--step', default=-1, type=int)

parser.add_argument('--n_seg', default=1, type=int)
parser.add_argument('--seq_len', default=1, type=int)

parser.add_argument('--max_step', default=3570, type=int)

parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--lr', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--lr_decay_mode', default='step', type=str)
parser.add_argument('--lr_decay_step', nargs='+', default=-1, type=int)

parser.add_argument('--lr_decay_gamma', default=0.1, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--grad_clip', default=100, type=float)

parser.add_argument('--acc_grad', default=0, type=int)

parser.add_argument('--save_every_step', default=700, type=float)
parser.add_argument('--update_every_step', default=1, type=int)

parser.add_argument('--evaluate', action='store_true')

parser.add_argument('--split', default='0', type=str)

args = parser.parse_args()
if args.run_id == '':
    args.run_id = id_generator()
    print('run_id: {}'.format(args.run_id))
if args.update_every_step == -1:
    args.update_every_step = 128/args.batch_size

if args.max_step == -1:
    args.max_step = 3500 # doubled
else:
    args.max_step = args.max_step * args.update_every_step
if args.lr_decay_step == -1:
    args.lr_decay_step = [1500, 3000]
else:
    args.lr_decay_step = [i * args.update_every_step for i in args.lr_decay_step]

#args.save_every_step = max(args.max_step/100, 1) # doubled

args.gpu_id = args.gpu_id.split(',')
args.gpu_id = [int(r) for r in args.gpu_id]


torch.cuda.set_device(args.gpu_id[0])

from torchvision import transforms
from torch.utils.data import DataLoader

from torch import optim
from torch.autograd.variable import Variable
from saver import Saver, make_log_dirs

import json

from tensorboardX import SummaryWriter


best_prec1 = 0
step = 0

def main():
    global step, best_prec1, model, crit, optimizer
    global train_dataset, test_dataset, train_loader, test_loader, vocab

    print('prepare dataset...')
    (train_dataset, train_loader), (test_dataset, test_loader) = prepare_dataset()

    # prepare model
    
    model, crit, optimizer = prepare_model()

    print('start training...')
    if args.evaluate:
        validate(True)
    else:
        train()


def prepare_dataset():    
    train_dataset = ImageDataProvider_mat("../Paper_data/train_elips.mat",is_flipping=False)

    test_dataset = ImageDataProvider_mat("../Paper_data/test_elips.mat",shuffle_data=False,is_flipping=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker,
                              pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker,
                              pin_memory=False)


    return (train_dataset, train_loader), (test_dataset, test_loader)

def prepare_model():
    from collections import OrderedDict
    model = UNet(1, depth=5, merge_mode='concat')

    crit = torch.nn.L1Loss()

    model = model.cuda(args.gpu_id[0])

    optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    return model, crit, optimizer

import subprocess, os, sys, datetime


def check_for_nan(ps):
    for p in ps:
        if p is not None:
            if not np.isfinite(np.sum(p.data.cpu().numpy())):
                return True
    return False

def train():
    global step, best_prec1
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    #
    end = time.time()
    model.train()
    for iter_, (imgs, labels) in enumerate(train_loader):
        target = labels.cuda(args.gpu_id[0])
        imgs = imgs.cuda(args.gpu_id[0])
        data_time.update(time.time()-end)

        target_var = Variable(target).type(torch.cuda.FloatTensor)
        imgs_var = Variable(imgs).type(torch.cuda.FloatTensor)

        # forward
        ys = torch.nn.parallel.data_parallel(model, imgs_var, args.gpu_id)
        loss = crit(ys, target_var)

        loss_meter.update(loss.data[0], len(imgs))

        # back
        # torch.autograd.backward([loss], [loss.data.new(1).fill_(1.0)])
        # loss_seq = [loss]
        # grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
        # torch.autograd.backward(loss_seq, grad_seq)
        loss.backward()

        # compute gradient and do SGD step
        if args.lr_decay_mode == 'poly':
            lr = poly_adjust_learning_rate(optimizer=optimizer, lr0=args.lr, step=step, n_step=args.max_step)

        elif args.lr_decay_mode == 'step':
            lr = step_adjust_learning_rate(optimizer=optimizer, lr0=args.lr, step=step, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

        elif args.lr_decay_mode == 'fix':
            lr = args.lr
        else:
            raise ValueError('lr_decay_mode wrong')

        total_norm = 0
        if (step % args.update_every_step == 0 and args.acc_grad) or (not args.acc_grad):
            total_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        batch_time.update(time.time() - end)

        epoch = step / max(train_loader.dataset.n_sample/args.batch_size, 1)
        if step % 10 == 0:
            print('time: {time} \t run_id: {run_id}\t'
                  'Epoch: [{0}][{1}/{2}/{step}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'lr {lr:.6f}\t'
                  'd_norm {total_norm:.3f}'.format(
                epoch, step, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=loss_meter, lr=lr, run_id=args.run_id, step=step,
                time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), total_norm=total_norm))

        # validate
        if (step % args.save_every_step == 0 or step >= args.max_step-1):
            loss = validate()
            print('Test loss: {loss.val:.4f}'.format(loss=loss_meter))

        if step >= args.max_step:
            break

        step += 1
        if check_for_nan(model.parameters()):
            print('nan in parameters')
            sys.exit(-1)
        end = time.time()
    return step

def validate(is_test=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    #
    model.eval()

    end = time.time()
    ys_all = []
    vid_paths_all = []
    model.eval()
    for iter_, (imgs, labels) in enumerate(test_loader):
        data_time.update(time.time()-end)
        target = labels.cuda(args.gpu_id[0])
        imgs = imgs.cuda(args.gpu_id[0])

        target_var = Variable(target)
        imgs_var = Variable(imgs, volatile=True)

        # forward
        ys = torch.nn.parallel.data_parallel(model, imgs_var, args.gpu_id)

        loss = crit(ys, target_var)

        loss_meter.update(loss.data[0], len(imgs))

        batch_time.update(time.time() - end)
        end = time.time()

        if iter_ % 1 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                iter_, len(test_loader), batch_time=batch_time, loss=loss_meter))
    model.train()
    return loss_meter.avg

def poly_adjust_learning_rate(optimizer, lr0, step, n_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr0 * (1.0 - step*1.0/n_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def step_adjust_learning_rate(optimizer, lr0, step, step_size, gamma):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if len(step_size) == 0:
        lr = lr0 * (gamma ** (step // step_size))
    else:
        lr = lr0 * gamma ** (sum([step > i for i in step_size]))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def my_save_checkpoint(saver, model, info_dict, is_best, step):
    saver.save(model=model, info_dict=info_dict, is_best=is_best, step=step)

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

if __name__ == '__main__':
    main()
