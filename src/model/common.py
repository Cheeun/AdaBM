import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import quantize
import numpy as np

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class ShortCut(nn.Module):
    def __init__(self):
        super(ShortCut, self).__init__()

    def forward(self, input):
        return input

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(self, args, conv, n_feats, kernel_size, bias=True, bn=False, act='relu', res_scale=1):
        super(ResBlock, self).__init__()

        self.args = args

        self.conv1 = quantize.QConv2d(args, n_feats, n_feats, kernel_size, bias=bias, non_adaptive=False)
        self.conv2 = quantize.QConv2d(args, n_feats, n_feats, kernel_size, bias=bias, non_adaptive=False)
        
        if act == 'relu':
            self.act = nn.ReLU(True)
        elif act== 'prelu':
            self.act = nn.PReLU()

        self.res_scale = res_scale
        self.is_bn = bn
        if bn:
            self.bn1 = nn.BatchNorm2d(n_feats)
            self.bn2 = nn.BatchNorm2d(n_feats)
        self.shortcut = ShortCut()

    def forward(self, x):
        if self.args.imgwise:
            bit_img = x[3]
        bit = x[2]
        feat = x[1]
        x = x[0]
        
        residual = self.shortcut(x)

        if self.args.imgwise:
            out,bit = self.conv1([x, bit, bit_img])
        else:
            out,bit = self.conv1([x, bit])

        if self.is_bn: 
            out = self.bn1(out)

        out1 = self.act(out)

        if self.args.imgwise:
            res, bit = self.conv2([out1, bit, bit_img])
        else:
            res, bit = self.conv2([out1,bit])
        
        if self.is_bn: 
            res = self.bn2(res)

        res = res.mul(self.res_scale)
        res += residual

        if feat is None:
            feat = res / torch.norm(res, p=2)
        else:
            feat = torch.cat([feat,  res / torch.norm(res, p=2)]) 

        if self.args.imgwise:
            return [res, feat, bit, bit_img]
        else:
            return [res, feat, bit]


class Upsampler(nn.Sequential):
    def __init__(self, args, conv, scale, n_feats, bn=False, act=False, bias=True, fq=False):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                if fq:
                    m.append(conv(args, n_feats, 4*n_feats, 3, bias=bias, non_adaptive=True, to_8bit=True))
                else:
                    m.append(conv(n_feats, 4 * n_feats, 3, bias=bias))

                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU())

        elif scale == 3:
            if fq:
                m.append(conv(args, n_feats, 9 * n_feats, 3, bias=bias, non_adaptive=True, to_8bit=True))
            else:
                m.append(conv(n_feats, 9 * n_feats, 3, bias=bias))

            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)