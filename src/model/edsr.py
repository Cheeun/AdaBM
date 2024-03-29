from model import common
from model import quantize

import torch.nn as nn
import torch

import kornia as K
import numpy as np

import cv2
import math

def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = 'relu'

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        self.fq = args.fq


        # Head module
        if args.fq:
            m_head = [quantize.QConv2d(args, args.n_colors, n_feats, kernel_size, bias=True, non_adaptive=True, to_8bit=True)]
        else:
            m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # Body module
        m_body = [
            common.ResBlock(
                args, conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(args.n_resblocks)
        ]

        if args.fq:
            m_body.append(quantize.QConv2d(args, n_feats, n_feats, kernel_size, bias=True))
        else:
            m_body.append(conv(n_feats, n_feats, kernel_size))

        # Tail module
        if args.fq:
            m_tail = [
                common.Upsampler(args, quantize.QConv2d, scale, n_feats, act=False, fq=args.fq), 
                quantize.QConv2d(args, n_feats, args.n_colors, kernel_size, bias=True, non_adaptive=True, to_8bit=True)
            ]
        else:
            m_tail = [
                common.Upsampler(args, conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        if args.imgwise:
            self.measure_l = nn.Parameter(torch.FloatTensor([128]).cuda())
            self.measure_u = nn.Parameter(torch.FloatTensor([128]).cuda())
            self.tanh = nn.Tanh()
            self.ema_epoch = 1
            self.init = False

        self.args = args


    def forward(self, x):
        if self.args.imgwise:
            image = x
            grads: torch.Tensor = K.filters.spatial_gradient(K.color.rgb_to_grayscale(image/255.), order=1)
            image_grad = torch.mean(torch.abs(grads.squeeze(1)),(1,2,3)) *1e+3 

            if self.init:
                # print(image_grad)
                if self.ema_epoch == 1:
                    measure_l = torch.quantile(image_grad.detach(), self.args.img_percentile/100.0)
                    measure_u = torch.quantile(image_grad.detach(), 1-self.args.img_percentile/100.0)
                    nn.init.constant_(self.measure_l, measure_l)
                    nn.init.constant_(self.measure_u, measure_u)
                else:
                    beta = self.args.ema_beta
                    new_measure_l = self.measure_l * beta + torch.quantile(image_grad.detach(), self.args.img_percentile/100.0) * (1-beta) 
                    new_measure_u = self.measure_u * beta + torch.quantile(image_grad.detach(), 1-self.args.img_percentile/100.0) * (1-beta) 
                    nn.init.constant_(self.measure_l, new_measure_l.item())
                    nn.init.constant_(self.measure_u, new_measure_u.item())

                self.ema_epoch += 1
                bit_img = torch.Tensor([0.0]).cuda()

            else:
                bit_img_soft = (image_grad - (self.measure_u + self.measure_l)/2) * (2/(self.measure_u - self.measure_l))
                bit_img_soft = self.tanh(bit_img_soft)
                bit_img_hard = (image_grad < self.measure_l) * (-1.0) + (image_grad >= self.measure_l) * (image_grad <= self.measure_u) * (0.0) + (image_grad> self.measure_u) *(1.0) 
                bit_img = bit_img_soft - bit_img_soft.detach() + bit_img_hard.detach() # the order matters
                bit_img = bit_img.view(bit_img.shape[0], 1, 1, 1)
            
        x = self.sub_mean(x)
        
        if self.args.fq:
            bit_fq = torch.zeros(x.shape[0]).cuda()
            x, bit_fq= self.head([x, bit_fq])

        else:
            x = self.head(x)
 
        feat = None
        bit = torch.zeros(x.shape[0]).cuda()

        res = x

        if self.args.imgwise:
            res, feat, bit, bit_img = self.body[:-1]([res, feat, bit, bit_img])
        else:
            res, feat, bit = self.body[:-1]([res, feat, bit])

        
        if self.args.fq:
            if self.args.imgwise:
                res, bit = self.body[-1:]([res, bit, bit_img])
            else:
                res, bit = self.body[-1:]([res, bit])
        else:
            res = self.body[-1:](res)

        res += x

        if self.args.fq:
            res, bit_fq = self.tail[0][0]([res, bit_fq]) # conv
            res = self.tail[0][1](res) # ps
            if len(self.tail[0]) == 4:
                res, bit_fq = self.tail[0][2]([res, bit_fq]) # conv 
                res = self.tail[0][3](res) # ps
            x, bit_fq = self.tail[-1]([res, bit_fq]) # conv
        else:
            x = self.tail(res)
            
        x = self.add_mean(x)


        return x, feat, bit
    

