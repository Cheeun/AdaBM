import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import time

class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in):
        x_out = torch.round(x_in)
        return x_out
    @staticmethod
    def backward(ctx, g):
        return g, None

class QConv2d(nn.Conv2d):
    def __init__(self, args, in_channels, out_channels, kernel_size, stride=1, 
                padding=1, bias=False, dilation=1, groups=1, non_adaptive=False, to_8bit=False):
        super(QConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation, groups=groups)
        self.args = args
        if not bias: self.bias = None
        self.dilation = (dilation, dilation)

        # For quantizing activations
        self.lower_a = nn.Parameter(torch.FloatTensor([-128]).cuda())
        self.upper_a = nn.Parameter(torch.FloatTensor([128]).cuda())
        self.round_a = Round.apply
        self.a_bit = self.args.quantize_a
        
        # For quantizing weights
        self.upper_w = nn.Parameter(torch.FloatTensor([128]).cuda())
        self.round_w = Round.apply
        self.w_bit = self.args.quantize_w

        self.non_adaptive = non_adaptive
        self.to_8bit = to_8bit

        if self.to_8bit:
            self.w_bit = 8.0
            self.a_bit = 8.0

        if not self.non_adaptive and self.args.layerwise:
            self.std_layer = []
            self.measure_layer = nn.Parameter(torch.FloatTensor([0]).cuda())
            self.tanh = nn.Tanh()

        self.ema_epoch = 1
        self.bac_epoch = 1
        self.init = False

    def init_qparams_a(self, x, quantizer=None):
        # Obtain statistics
        if quantizer == 'minmax':
            lower_a = torch.min(x).detach().cpu()
            upper_a = torch.max(x).detach().cpu()

        elif quantizer == 'percentile':
            try:
                lower_a = torch.quantile(x.reshape(-1), 1.0-self.args.percentile_alpha).detach().cpu()
                upper_a = torch.quantile(x.reshape(-1), self.args.percentile_alpha).detach().cpu()
            except:
                lower_a = np.percentile(x.reshape(-1).detach().cpu(), (1.0-self.args.percentile_alpha)*100.0)
                upper_a = np.percentile(x.reshape(-1).detach().cpu(), self.args.percentile_alpha*100.0)

        elif quantizer == 'omse':
            lower_a = torch.min(x).detach()
            upper_a = torch.max(x).detach()
            best_score = 1e+10
            for i in range(90):
                new_lower = lower_a * (1.0 - i*0.01)
                new_upper = upper_a * (1.0 - i*0.01)
                x_q = torch.clamp(x, min= new_lower, max=new_upper)
                x_q = (x_q - new_lower) / (new_upper - new_lower)
                x_q = torch.round(x_q * (2**self.args.quantize_a -1)) / (2**self.args.quantize_a -1)
                x_q = x_q * (new_upper - new_lower) + new_lower
                score = (x - x_q).abs().pow(2.0).mean()
                if score < best_score:
                    best_score = score
                    best_lower = new_lower
                    best_upper = new_upper
            lower = best_lower.cpu()
            upper = best_upper.cpu()
    
        # Update q params
        if self.ema_epoch == 1:
            nn.init.constant_(self.lower_a, lower_a)
            nn.init.constant_(self.upper_a, upper_a)
        else:
            beta = self.args.ema_beta
            lower_a = lower_a * (1-beta) + self.lower_a  * beta
            upper_a = upper_a * (1-beta) + self.upper_a  * beta
            nn.init.constant_(self.lower_a, lower_a.item())
            nn.init.constant_(self.upper_a, upper_a.item())
    

    def init_qparams_w(self, w, quantizer=None):
        if quantizer == 'minmax':
            upper_w = torch.max(torch.abs(self.weight)).detach()
        elif quantizer == 'percentile':
            try:
                upper_w = torch.quantile(self.weight.reshape(-1), self.args.percentile_alpha).detach().cpu()
            except:
                upper_w = np.percentile(self.weight.reshape(-1).detach().cpu(), self.args.percentile_alpha*100.0)
        elif quantizer == 'omse':
            upper_w = torch.max(self.weight).detach()
            best_score_w = 1e+10
            for i in range(50):
                new_upper_w = upper_w * (1.0 - i*0.01)
                w_q = torch.clamp(self.weight, min=-new_upper_w, max=new_upper_w).detach()
                w_q = (w_q + new_upper_w) / (2*new_upper_w )
                w_q = torch.round(w_q * (2**self.args.quantize_w -1)) / (2**self.args.quantize_w -1)
                w_q = w_q * (2*new_upper_w) - new_upper_w
                score = (self.weight - w_q).abs().pow(2.0).mean().detach().cpu()
                if score < best_score_w:
                    best_score_w = score
                    best_i = i
                    upper = new_upper_w
            upper_w = upper.cpu()
        
        nn.init.constant_(self.upper_w, upper_w)
        
    def forward(self,x):
        if self.args.imgwise and not self.non_adaptive:
            bit_img = x[2]
        bit = x[1]
        x = x[0]

        if self.w_bit == 32:
            if self.init:
                # Initialize q params
                self.init_qparams_a(x, quantizer=self.args.quantizer)
                if self.ema_epoch == 1:
                    self.init_qparams_w(self.weight, quantizer=self.args.quantizer_w)

                if not self.non_adaptive and self.args.layerwise:
                    measure = lambda x: torch.mean(torch.std(x.detach(), dim=(1,2,3,)))
                    measure_layer = measure(x)
                    self.std_layer.append(measure_layer.detach().cpu().numpy())

                self.ema_epoch += 1

            a_bit = torch.Tensor([32.0]).cuda()
            w = self.weight

        else:
            # Obtain bit-width
            if not self.non_adaptive and (self.args.imgwise or self.args.layerwise):
                a_bit = self.a_bit
                if self.args.imgwise:
                    a_bit += bit_img
                if self.args.layerwise:                                        
                    bit_layer_hard = torch.round(torch.clamp(self.measure_layer, min=-1.0, max=1.0))
                    bit_layer_soft = self.tanh(self.measure_layer) 
                    bit_layer = bit_layer_soft - bit_layer_soft.detach() + bit_layer_hard.detach()
                    a_bit += bit_layer
            else:
                a_bit = torch.tensor(self.a_bit).repeat(x.shape[0], 1, 1, 1).cuda()

            # Bit-aware Clipping
            if self.args.bac:
                do_bac = self.bac_epoch == 1
                # Do BaC after init phase ends
                if do_bac:
                    self.bac_epoch += 1
                    if self.training and not self.to_8bit:
                        best_score = 1e+10
                        lower_a = self.lower_a
                        upper_a = self.upper_a

                        for i in range(100):
                            new_lower_a = self.lower_a * (1.0 - i*0.01)
                            new_upper_a = self.upper_a * (1.0 - i*0.01)
                            x_q_temp = torch.clamp(x.clone().detach(), min= new_lower_a, max=new_upper_a)
                            x_q_temp = (x_q_temp - new_lower_a) / (new_upper_a - new_lower_a)
                            if not self.non_adaptive and self.args.layerwise:
                                x_q_temp = torch.round(x_q_temp * (2**(self.a_bit+bit_layer_hard) -1)) / (2**(self.a_bit+bit_layer_hard) -1)
                            else:
                                x_q_temp = torch.round(x_q_temp * (2**(self.a_bit) -1)) / (2**(self.a_bit) -1)
                                
                            x_q_temp = x_q_temp * (new_upper_a - new_lower_a) + new_lower_a
                            score = (x.clone().detach() - x_q_temp).abs().pow(2.0).mean()
                            if score < best_score:
                                best_i = i
                                best_score = score
                                lower_a = new_lower_a
                                upper_a = new_upper_a
                        
                        new_lower = self.lower_a * self.args.bac_beta + lower_a * (1-self.args.bac_beta)
                        new_upper = self.upper_a * self.args.bac_beta + upper_a * (1-self.args.bac_beta)
                        nn.init.constant_(self.lower_a, new_lower.item())
                        nn.init.constant_(self.upper_a, new_upper.item())

            # Quantize activations
            x_c = torch.clamp(x, min=self.lower_a, max=self.upper_a)
            x_c2 = (x_c - self.lower_a) / (self.upper_a - self.lower_a)
            x_c2 = x_c2 * (2**a_bit -1)
            x_int = self.round_a(x_c2)
            x_int = x_int / (2**a_bit -1)
            x_q = x_int * (self.upper_a - self.lower_a) + self.lower_a
            x = x_q

            # Quantize weights
            w_c = torch.clamp(self.weight, min=-self.upper_w, max=self.upper_w)
            w_c2 = (w_c + self.upper_w) / (2*self.upper_w )
            w_c2 = (w_c2) * (2**self.w_bit-1)
            w_int = self.round_w(w_c2)
            w_int = w_int / (2**self.w_bit-1)
            w_q = w_int * (2*self.upper_w) - self.upper_w
            w = w_q

        self.padding = (self.kernel_size[0]//2, self.kernel_size[1]//2)        
        out = F.conv2d(x, w, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        bit += a_bit.view(-1)

        return out, bit