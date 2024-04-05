import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from pathlib import Path
import shutil
import torch.nn as nn
import torch.nn.functional as F
import logging
import coloredlogs
import cv2
import functools
from torchvision.utils import make_grid
from decimal import Decimal
from math import exp


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def accumulate(self, val, n=1):
        self.sum += val
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        self.ssim_log = torch.Tensor()
        self.bit_log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            # self.dir = os.path.join('..', 'experiment', '{}_sd{}'.format(args.save, args.seed))
            self.dir = os.path.join('..', 'experiment', '{}'.format(args.save))
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)
        if args.test_own is not None:
            os.makedirs(self.get_path('results-test_own'), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        
        # self.plot_bit(epoch)
        # self.plot_psnr(epoch)

    def add_log(self, log):
        self.log = torch.cat([self.log, log])
        self.ssim_log = torch.cat([self.ssim_log, log])
        self.bit_log = torch.cat([self.bit_log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        # axis = np.linspace(1, epoch, epoch)
        axis = np.linspace(1, epoch-1, epoch-1)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.png'.format(d)))

            plt.close(fig)

    def plot_bit(self, epoch):
        # axis = np.linspace(1, epoch, epoch)
        axis = np.linspace(1, epoch-1, epoch-1)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.bit_log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Average Bit')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}_bit.png'.format(d)))

            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list):
        if self.args.save_results:
            if isinstance(dataset, str):
                filename = self.get_path(
                    'results-{}'.format(dataset),
                    '{}'.format(filename)
                )
            else: 
                filename = self.get_path(
                    'results-{}'.format(dataset.dataset.name),
                    '{}'.format(filename)
                )

            postfix = ('', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window_3d(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous().cuda()
    return window

def ssim_matlab(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, _, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(real_size, channel=1).to(img1.device)
        # Channel is set to 1 since we consider color images as volumetric images

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu2_sq
    sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
            sr = sr.mul(convert).sum(dim=1)
            hr = hr.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    sr = sr[..., shave:-shave, shave:-shave]
    hr = hr[..., shave:-shave, shave:-shave] 

    ssim_out = ssim_matlab(sr.unsqueeze(0), hr.unsqueeze(0), val_range=255).item()

    return -10 * math.log10(mse), ssim_out


def crop(img, crop_sz, step):
    b, c, h, w = img.shape
    h_space = np.arange(0, max(h - crop_sz,0) + 1, step)
    w_space = np.arange(0, max(w - crop_sz,0) + 1, step)
    num_h = 0
    lr_list=[]
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            # remaining borders are NOT clipped
            x_end = x + crop_sz if x != h_space[-1] else h
            y_end = y + crop_sz if y != w_space[-1] else w

            crop_img = img[:,:, x : x_end, y : y_end]
            lr_list.append(crop_img)

    return lr_list, num_h, num_w, h, w

def crop_parallel(img, crop_sz, step):
    # remaining borders are clipped
    b, c, h, w = img.shape
    h_space = np.arange(0, h - crop_sz + 1, step)
    w_space = np.arange(0, w - crop_sz + 1, step)
    index = 0
    num_h = 0
    lr_list=torch.Tensor().to(img.device)
    for x in h_space:
        num_h += 1
        num_w = 0
        for y in w_space:
            num_w += 1
            index += 1
            crop_img = img[:, :, x:x + crop_sz, y:y + crop_sz]
            lr_list = torch.cat([lr_list, crop_img])
    new_h=x + crop_sz # new height after crop
    new_w=y + crop_sz # new width  after crop
    return lr_list, num_h, num_w, new_h, new_w


def combine(sr_list, num_h, num_w, h, w, patch_size, step, scale):
    index=0
    sr_img = torch.zeros((1, 3, h*scale, w*scale)).cuda()
    step = step * scale
    patch_size = patch_size * scale

    for x in range(num_h):
        for y in range(num_w):
            x_patch_size = sr_list[index].shape[2]
            y_patch_size = sr_list[index].shape[3]
            sr_img[:, :, x*step : x*step+x_patch_size, y*step : y*step+y_patch_size] += sr_list[index]
            index += 1

    # mean the overlap region
    for x in range(1, num_h):
        sr_img[:, :, x*step : x*step+ (patch_size - step), :]/=2
    for y in range(1, num_w):
        sr_img[:, :, :, y*step : y*step+ (patch_size - step)]/=2

    return sr_img

