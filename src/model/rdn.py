from model import common
from model import quantize
import torch
import torch.nn as nn
import kornia as K

def make_model(args, parent=False):
    return RDN(args)

class RDB_Conv(nn.Module):
    def __init__(self, args, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        bias = True
        self.args = args
        self.conv = nn.Sequential(*[
            quantize.QConv2d(args, Cin, G, kSize, stride=1, padding=(kSize-1)//2, bias=True, dilation=1, groups=1, non_adaptive=False),
            nn.ReLU()
        ])

    def forward(self, x):
        if self.args.imgwise:
            bit_img = x[2]
        bit = x[1]
        x = x[0]
        out = x

        if self.args.imgwise:
            out, bit = self.conv[0]([out, bit, bit_img])
        else:
            out, bit = self.conv[0]([out, bit])

        out = self.conv[1](out)
        out = torch.cat((x, out), 1)

        if self.args.imgwise:
            return [out, bit, bit_img]
        else:
            return [out, bit]


class RDB(nn.Module):
    def __init__(self, args, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(args, G0 + c*G, G, kSize))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = quantize.QConv2d(args, G0 + C*G, G0, 1, padding=0, stride=1, bias=True, non_adaptive=True, to_8bit=False) # 1x1 conv is non-adaptively quantized
        self.args = args

    def forward(self, x):
        if self.args.imgwise:
            bit_img = x[3]
        bit = x[2]
        feat = x[1]
        x = x[0]

        out = x
        if self.args.imgwise:
            out, bit, bit_img = self.convs([out, bit, bit_img])
        else:
            out, bit = self.convs([out, bit])

        out, bit = self.LFF([out, bit])
        out += x 

        feat_ = out / torch.norm(out, p=2) / (out.shape[1]*out.shape[2]*out.shape[3])
        if feat is None:
            feat = feat_
        else:
            feat = torch.cat([feat, feat_])
        
        if self.args.imgwise:
            return out, feat, bit, bit_img
        else:
            return out, feat, bit

class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        self.args = args

        # number of RDB blocks, conv layers, out channels
        self.D, self.C, self.G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        # Shallow feature extraction net
        if args.fq:
            self.SFENet1 = quantize.QConv2d(args, args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1, bias=True, non_adaptive=True, to_8bit=True)
            self.SFENet2 = quantize.QConv2d(args, G0, G0, kSize, padding=(kSize-1)//2, stride=1, bias=True, non_adaptive=True, to_8bit=True)
        else:
            self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
            self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(args, growRate0 = G0, growRate = self.G, nConvLayers = self.C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            quantize.QConv2d(args, self.D * G0, G0, 1, padding=0, stride=1, bias=True),
            quantize.QConv2d(args, G0, G0, kSize, padding=(kSize-1)//2, stride=1, bias=True)
        ])

        # Up-sampling net
        if args.fq:
            if r == 2 or r == 3:
                self.UPNet = nn.Sequential(*[
                    quantize.QConv2d(args, G0, self.G * r * r, kSize, padding=(kSize-1)//2, stride=1, bias=True, non_adaptive=True, to_8bit=True),
                    nn.PixelShuffle(r),
                    quantize.QConv2d(args, self.G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1, bias=True, non_adaptive=True, to_8bit=True)
                ])
            elif r == 4:
                self.UPNet = nn.Sequential(*[
                    quantize.QConv2d(args, G0, self.G * 4, kSize, padding=(kSize-1)//2, stride=1, bias=True, non_adaptive=True, to_8bit=True),
                    nn.PixelShuffle(2),
                    quantize.QConv2d(args, self.G, self.G * 4, kSize, padding=(kSize-1)//2, stride=1, bias=True, non_adaptive=True, to_8bit=True),
                    nn.PixelShuffle(2),
                    quantize.QConv2d(args, self.G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1, bias=True, non_adaptive=True, to_8bit=True)
                ])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")
        
        else:
            if r == 2 or r == 3:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, self.G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(r),
                    nn.Conv2d(self.G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            elif r == 4:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, self.G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(self.G, self.G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(self.G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")
        
        if args.imgwise:
            self.measure_l = nn.Parameter(torch.FloatTensor([128]).cuda())
            self.measure_u = nn.Parameter(torch.FloatTensor([128]).cuda())
            self.tanh = nn.Tanh()
            self.ema_epoch = 1
            self.init = False

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
                bit_img = bit_img_soft - bit_img_soft.detach() + bit_img_hard.detach()# the order matters
                bit_img = bit_img.view(bit_img.shape[0], 1, 1, 1)

        feat = None; bit = torch.zeros(x.shape[0]).cuda(); bit_fq = torch.zeros(x.shape[0]).cuda()
        
        if self.args.fq:
            f__1, bit_fq = self.SFENet1([x, bit_fq])
            x, bit_fq  = self.SFENet2([f__1, bit_fq])
        else:
            f__1 = self.SFENet1(x)
            x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            if self.args.imgwise:
                x, feat, bit, bit_img = self.RDBs[i]([x, feat, bit, bit_img])
            else:
                x, feat, bit = self.RDBs[i]([x, feat, bit])
            RDBs_out.append(x)

        if self.args.imgwise:
            x, bit = self.GFF[0]([torch.cat(RDBs_out,1), bit, bit_img])
            x, bit = self.GFF[1]([x, bit, bit_img])
        else:
            x, bit = self.GFF[0]([torch.cat(RDBs_out,1), bit])
            x, bit = self.GFF[1]([x, bit])

        x += f__1
    
        if self.args.fq:
            out, bit_fq = self.UPNet[0]([x, bit_fq])
            out = self.UPNet[1](out)
            out, bit_fq = self.UPNet[2]([out, bit_fq])
            if len(self.UPNet) > 3:
                out = self.UPNet[3](out)
                out, bit_fq = self.UPNet[4]([out, bit_fq])
        else:
            out = self.UPNet(x)

        return out, feat, bit
