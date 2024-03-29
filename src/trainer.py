import os
import glob
import sys
import math
import time
import datetime
import shutil

import numpy as np

import torch
import torch.nn.utils as utils
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torchvision.utils import save_image

from decimal import Decimal
from tqdm import tqdm
import cv2

import utility
from model.quantize import QConv2d
import kornia as K

class Trainer():
    def __init__(self, args, loader, my_model, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_init = loader.loader_init
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.model = my_model
        self.epoch = 0 

        shutil.copyfile('./trainer.py', os.path.join(self.ckp.dir, 'trainer.py'))
        shutil.copyfile('./model/quantize.py', os.path.join(self.ckp.dir, 'quantize.py'))

        quant_params_a = [v for k, v in self.model.model.named_parameters() if '_a' in k]
        quant_params_w = [v for k, v in self.model.model.named_parameters() if '_w' in k]

        if args.layerwise or args.imgwise:
            quant_params_measure= []
            if args.layerwise:
                quant_params_measure_layer = [v for k, v in self.model.model.named_parameters() if 'measure_layer' in k]
                quant_params_measure.append({'params': quant_params_measure_layer, 'lr': args.lr_measure_layer})
            if args.imgwise:
                quant_params_measure_image = [v for k, v in self.model.model.named_parameters() if 'measure' in k and 'measure_layer' not in k]
                quant_params_measure.append({'params': quant_params_measure_image, 'lr': args.lr_measure_img})
        
            self.optimizer_measure = torch.optim.Adam(quant_params_measure, betas=args.betas, eps=args.epsilon)
            self.scheduler_measure = lrs.StepLR(self.optimizer_measure, step_size=args.step, gamma=args.gamma)

        self.optimizer_a = torch.optim.Adam(quant_params_a, lr=args.lr_a, betas=args.betas, eps=args.epsilon)
        self.optimizer_w = torch.optim.Adam(quant_params_w, lr=args.lr_w, betas=args.betas, eps=args.epsilon)
        self.scheduler_a = lrs.StepLR(self.optimizer_a, step_size=args.step, gamma=args.gamma)
        self.scheduler_w = lrs.StepLR(self.optimizer_w, step_size=args.step, gamma=args.gamma)
        
        self.skt_losses = utility.AverageMeter()
        self.pix_losses = utility.AverageMeter()
        self.bit_losses = utility.AverageMeter()

        self.num_quant_modules = 0
        for n, m in self.model.named_modules():
            if isinstance(m, QConv2d):
                if not m.to_8bit: # 8-bit (first or last) modules are excluded for the bit count
                    self.num_quant_modules +=1
        
        # for initialization
        if not args.test_only:
            for n, m in self.model.named_modules():
                if isinstance(m, QConv2d):
                    setattr(m, 'w_bit', 32.0)
                    setattr(m, 'a_bit', 32.0)
                    setattr(m, 'init', True)
            if args.imgwise:
                setattr(self.model.model, 'init', True)
    
    def get_stage_optimizer_scheduler(self):
        if self.args.imgwise or self.args.layerwise:
            # w -> a -> measure
            if (self.epoch-1) % 3 == 0:
                param_name = '_w'
                optimizer = self.optimizer_w
                scheduler = self.scheduler_w
            elif (self.epoch-1) % 3 == 1:
                param_name = '_a'
                optimizer = self.optimizer_a
                scheduler = self.scheduler_a
            else:
                param_name = 'measure'
                optimizer = self.optimizer_measure
                scheduler = self.scheduler_measure
        else:
            if (self.epoch-1) % 2 == 0:
                param_name = '_w'
                optimizer = self.optimizer_w
                scheduler = self.scheduler_w
            else:
                param_name = '_a'
                optimizer = self.optimizer_a
                scheduler = self.scheduler_a

        return param_name, optimizer, scheduler
    
    def set_bit(self, teacher=False):
        for n, m in self.model.named_modules():
            if isinstance(m, QConv2d):
                if teacher:
                    setattr(m, 'w_bit', 32.0)
                    setattr(m, 'a_bit', 32.0)
                elif m.non_adaptive:
                    if m.to_8bit:
                        setattr(m, 'w_bit', 8.0)
                        setattr(m, 'a_bit', 8.0)
                    else:
                        setattr(m, 'w_bit', self.args.quantize_w)
                        setattr(m, 'a_bit', self.args.quantize_a)
                else:
                    setattr(m, 'w_bit', self.args.quantize_w)
                    setattr(m, 'a_bit', self.args.quantize_a)

                setattr(m, 'init', False)

        if self.args.imgwise:
            setattr(self.model.model, 'init', False)

        
    def train(self):
        if self.epoch > 0:
            param_name, optimizer, scheduler = self.get_stage_optimizer_scheduler()
            epoch_update = 'Update param ' + param_name
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            self.ckp.write_log(
                '\n[Epoch {}]\t {}\t Learning rate for param: {:.2e}'.format(
                    self.epoch,
                    epoch_update,
                    Decimal(lr))
            )
            
        self.model.train()
        
        timer_data, timer_model = utility.timer(), utility.timer()
        start_time = time.time()
        
        if self.epoch == 0:
            # Initialize Q parameters using freezed FP model
            params = self.model.named_parameters()
            for name1, params1 in params:
                params1.requires_grad=False

            for batch, (lr, _, idx_scale,) in enumerate(self.loader_init):
                lr, = self.prepare(lr)

                timer_data.hold()
                timer_model.tic()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    sr_temp, feat_temp, bit_temp = self.model(lr, idx_scale)
                display_bit = bit_temp.mean() / self.num_quant_modules
                
                if self.args.imgwise or self.args.layerwise:
                    self.ckp.write_log('[{}/{}] [bit:{:.2f}] \t{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.loader_init.batch_size,
                        len(self.loader_init.dataset),
                        display_bit,
                        timer_model.release(),
                        timer_data.release(), 
                    ))
            
            if self.args.layerwise:
                measure_layer_list = []
                for n, m in self.model.named_modules():
                    if isinstance(m, QConv2d) and not m.non_adaptive:
                        if hasattr(m, 'std_layer') and len(m.std_layer) > 0:
                            measure_layer_list.append(np.mean(m.std_layer))
                mu = np.mean(measure_layer_list)
                sig = np.std(measure_layer_list)

                lower = np.percentile(measure_layer_list, self.args.layer_percentile)
                upper = np.percentile(measure_layer_list, 100.0-self.args.layer_percentile)

                for n, m in self.model.named_modules():
                    if isinstance(m, QConv2d):
                        if not m.non_adaptive:
                            normalized_measure = (np.mean(m.std_layer) < lower) * (-1.0) + (np.mean(m.std_layer) > upper) * (1.0)  
                            torch.nn.init.constant_(m.measure_layer, normalized_measure)
                            m.std_layer.clear()

            print('Calibration done!')
            # self.set_bit(teacher=False)

            if self.args.imgwise:
                print('image-lower:{:.3f}, image-upper:{:.3f}'.format(
                    self.model.model.measure_l.data.item(),
                    self.model.model.measure_u.data.item(),
                    ))
            if self.args.layerwise:
                bit_layer_list = []
                for n, m in self.model.named_modules():
                    if isinstance(m, QConv2d):
                        if not m.non_adaptive:
                            bit_layer_list.append(int(m.measure_layer.data.item()))
                print(bit_layer_list)
                print(np.mean(bit_layer_list)) 
        else:
            # Update quantization parameters
            for k, v in self.model.named_parameters():
                if param_name in k:
                    v.requires_grad=True
                else:
                    v.requires_grad=False
            
            self.bit_losses.reset()
            self.pix_losses.reset()
            self.skt_losses.reset()

            for batch, (lr, _, idx_scale,) in enumerate(self.loader_train):
                lr, = self.prepare(lr)
                timer_data.hold()
                timer_model.tic()

                optimizer.zero_grad()

                with torch.no_grad():
                    self.set_bit(teacher=True)
                    sr_t, feat_t, bit_t = self.model(lr, idx_scale)
                
                self.set_bit(teacher=False)
                sr, feat, bit = self.model(lr, idx_scale) 

                loss = 0.0
                pix_loss = F.l1_loss(sr, sr_t)

                loss += pix_loss

                skt_loss = 0.0
                for block in range(len(feat)):
                    skt_loss += self.args.w_sktloss * torch.norm(feat[block]-feat_t[block], p=2) / sr.shape[0] / len(feat)
                loss += skt_loss

                if self.args.layerwise or self.args.imgwise:
                    average_bit = bit.mean() / self.num_quant_modules
                    bit_loss = self.args.w_bitloss* torch.max(average_bit-(self.args.quantize_a), torch.zeros_like(average_bit))
                    if param_name == 'measure':
                        loss +=  bit_loss
                
                loss.backward()

                self.pix_losses.update(pix_loss.item(), lr.size(0))
                display_pix_loss = f'L_pix: {self.pix_losses.avg: .3f}'
                self.skt_losses.update(skt_loss.item(), lr.size(0))
                display_skt_loss = f'L_skt: {self.skt_losses.avg: .3f}'

                if self.args.layerwise or self.args.imgwise:
                    self.bit_losses.update(bit_loss.item(), lr.size(0))
                    display_bit_loss = f'L_bit: {self.bit_losses.avg: .3f}'

                optimizer.step()
                timer_model.hold()

                if (batch + 1) % self.args.print_every == 0:
                    display_bit = bit.mean() / self.num_quant_modules
                    if self.args.layerwise or self.args.imgwise:
                        self.ckp.write_log('[{}/{}]\t{} \t{} \t{} [bit:{:.2f}] \t{:.1f}+{:.1f}s'.format(
                            (batch + 1) * self.loader_train.batch_size,
                            len(self.loader_train.dataset),
                            display_pix_loss,
                            display_skt_loss,
                            display_bit_loss,
                            display_bit,
                            timer_model.release(),
                            timer_data.release(), 
                        ))
                    else:
                        self.ckp.write_log('[{}/{}]\t{} \t{} [bit:{:.2f}] \t{:.1f}+{:.1f}s'.format(
                            (batch + 1) * self.loader_train.batch_size,
                            len(self.loader_train.dataset),
                            display_pix_loss,
                            display_skt_loss,
                            display_bit,
                            timer_model.release(),
                            timer_data.release(), 
                        ))
                timer_data.tic()

            scheduler.step()

        self.epoch += 1

        end_time = time.time()
        time_interval = end_time - start_time
        t_string = "Epoch Running Time is: " + str(datetime.timedelta(seconds=time_interval)) + "\n"
        self.ckp.write_log('{}'.format(t_string))
    
    def patch_inference(self, model, lr, idx_scale):
        patch_idx = 0
        tot_bit_image = 0
        if self.args.n_parallel!=1: 
            lr_list, num_h, num_w, h, w = utility.crop_parallel(lr, self.args.test_patch_size, self.args.test_step_size)
            sr_list = torch.Tensor().cuda()
            for lr_sub_index in range(len(lr_list)// self.args.n_parallel + 1):
                torch.cuda.empty_cache()
                with torch.no_grad():
                    sr_sub, feat, bit = self.model(lr_list[lr_sub_index* self.args.n_parallel: (lr_sub_index+1)*self.args.n_parallel], idx_scale)
                    sr_sub = utility.quantize(sr_sub, self.args.rgb_range)
                sr_list = torch.cat([sr_list, sr_sub])
                average_bit = bit.mean() / self.num_quant_modules
                tot_bit_image += average_bit
                patch_idx += 1
            sr = utility.combine(sr_list, num_h, num_w, h, w, self.args.test_patch_size, self.args.test_step_size, self.scale[0])
        else:
            lr_list, num_h, num_w, h, w = utility.crop(lr, self.args.test_patch_size, self.args.test_step_size)
            sr_list = []
            for lr_sub_img in lr_list:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    sr_sub, feat, bit = self.model(lr_sub_img, idx_scale)
                    sr_sub = utility.quantize(sr_sub, self.args.rgb_range)
                sr_list.append(sr_sub)
                average_bit = bit.mean() / self.num_quant_modules
                tot_bit_image += average_bit
                patch_idx += 1
            sr = utility.combine(sr_list, num_h, num_w, h, w, self.args.test_patch_size, self.args.test_step_size, self.scale[0])

        bit = tot_bit_image / patch_idx

        return sr, feat, bit

    def test(self):
        torch.set_grad_enabled(False)
        
        # if True:
        if self.epoch > 1 or self.args.test_only:
            self.ckp.write_log('\nEvaluation:')
            self.ckp.add_log(
                torch.zeros(1, len(self.loader_test), len(self.scale))
            )
            self.model.eval()
            timer_test = utility.timer()
        
            if self.epoch == 2 or self.args.test_only:
                ################### Num of Params, Storage Size ####################
                n_params = 0
                n_params_q = 0
                for k, v in self.model.named_parameters():
                    nn = np.prod(v.size())
                    n_params += nn

                    if 'weight' in k:
                        name_split = k.split(".")
                        del name_split[-1]
                        module_temp = self.model
                        for n in name_split:
                            module_temp = getattr(module_temp, n)
                        if isinstance(module_temp, QConv2d):
                            n_params_q += nn * module_temp.w_bit / 32.0
                            # print(k, module_temp.w_bit)
                        else:
                            n_params_q += nn
                    else:
                        n_params_q += nn

                self.ckp.write_log('Parameters: {:.3f}K'.format(n_params/(10**3)))
                self.ckp.write_log('Model Size: {:.3f}K'.format(n_params_q/(10**3)))
        
            if self.args.save_results:
                self.ckp.begin_background()
        
            ############################## TEST FOR OWN #############################
            if self.args.test_own is not None:
                test_img = cv2.imread(self.args.test_own)
                lr = torch.tensor(test_img).permute(2,0,1).float().cuda()
                lr = torch.flip(lr, (0,)) # for color
                lr = lr.unsqueeze(0)

                tot_bit = 0
                for idx_scale, scale in enumerate(self.scale):
                    if self.args.test_patch:
                        sr, feat, bit = self.patch_inference(self.model, lr, idx_scale)
                        img_bit = bit
                    else:
                        with torch.no_grad():
                            sr, feat, bit = self.model(lr, idx_scale)
                        img_bit = bit.mean() / self.num_quant_modules

                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]


                    filename = self.args.test_own.split('/')[-1].split('.')[0]
                    if self.args.save_results:
                        save_name = '{}_x{}_{:.2f}bit'.format(filename, scale, img_bit)
                        self.ckp.save_results('test_own', save_name, save_list)

                    self.ckp.write_log('[{} x{}] Average Bit: {:.2f} '.format(filename, scale, img_bit))

            ############################## TEST FOR TEST SET #############################
            if self.args.test_own is None:
                for idx_data, d in enumerate(self.loader_test):
                    for idx_scale, scale in enumerate(self.scale):
                        d.dataset.set_scale(idx_scale)
                        tot_ssim =0
                        tot_bit =0 
                        i=0
                        bitops =0
                        for lr, hr, filename in tqdm(d, ncols=80):
                            i+=1
                            lr, hr = self.prepare(lr, hr)
                        
                            if self.args.test_patch:
                                sr, feat, bit = self.patch_inference(self.model, lr, idx_scale)
                                if self.args.n_parallel!=1: hr = hr[:, :, :lr.shape[2]*self.scale[0], :lr.shape[2]*self.scale[0]] 
                                img_bit = bit.item()
                            else:
                                with torch.no_grad():
                                    sr, feat, bit = self.model(lr, idx_scale)
                                img_bit = bit.mean().item() / self.num_quant_modules


                            sr = utility.quantize(sr, self.args.rgb_range)
                            save_list = [sr]

                            psnr, ssim = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)

                            self.ckp.ssim_log[-1, idx_data, idx_scale] += ssim
                            self.ckp.log[-1, idx_data, idx_scale] += psnr
                            self.ckp.bit_log[-1, idx_data, idx_scale] += img_bit
                            
                            if self.args.save_gt:
                                save_list.extend([lr, hr])

                            if self.args.save_results:
                                save_name = '{}_x{}_{:.2f}dB_{:.2f}bit'.format(filename[0], scale, psnr, img_bit)
                                self.ckp.save_results(d, save_name, save_list, scale)
                            
                        self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                        self.ckp.ssim_log[-1, idx_data, idx_scale] /= len(d)
                        self.ckp.bit_log[-1, idx_data, idx_scale] /= len(d)

                        best = self.ckp.log.max(0)
                        self.ckp.write_log(
                            '[{} x{}]\tPSNR: {:.3f} \t SSIM: {:.4f} \tBit: {:.2f} \t(Best: {:.3f} @epoch {})'.format(
                                d.dataset.name,
                                scale,
                                self.ckp.log[-1, idx_data, idx_scale],
                                self.ckp.ssim_log[-1, idx_data, idx_scale],
                                self.ckp.bit_log[-1, idx_data, idx_scale],
                                best[0][idx_data, idx_scale],
                                best[1][idx_data, idx_scale] + 1
                            )
                        )
                        
            if self.args.save_results:
                self.ckp.end_background()
            
            # save models
            if not self.args.test_only:
                self.ckp.save(self, self.epoch, is_best=(best[1][0, 0] + 1 == self.epoch -1))

        torch.set_grad_enabled(True) 

    def test_teacher(self):
        torch.set_grad_enabled(False)
        self.model.eval()
        self.ckp.write_log('Teacher Evaluation')

        ############################## Num of Params ####################
        n_params = 0
        for k, v in self.model.named_parameters():
            if '_a' not in k and '_w' not in k and 'measure' not in k: # for teacher model
                n_params += np.prod(v.size())
        self.ckp.write_log('Parameters: {:.3f}K'.format(n_params/(10**3)))

        if self.args.save_results:
            self.ckp.begin_background()
        
        ############################## TEST FOR OWN #############################
        if self.args.test_own is not None:
            test_img = cv2.imread(self.args.test_own)
            lr = torch.tensor(test_img).permute(2,0,1).float().cuda()
            lr = torch.flip(lr, (0,)) # for color
            lr = lr.unsqueeze(0)

            tot_bit = 0
            for idx_scale, scale in enumerate(self.scale):
                self.set_bit(teacher=True)
                if self.args.test_patch:
                    sr, feat, bit = self.patch_inference(self.model, lr, idx_scale)
                    img_bit = bit
                else:
                    with torch.no_grad():
                        sr, feat, bit = self.model(lr, idx_scale)
                    img_bit = bit.mean() / self.num_quant_modules                    
                self.set_bit(teacher=False)

                sr = utility.quantize(sr, self.args.rgb_range)
                save_list = [sr]
                if self.args.save_results:
                    filename = self.args.test_own.split('/')[-1].split('.')[0]
                    save_name = '{}_x{}_{:.2f}bit'.format(filename, scale, img_bit)
                    self.ckp.save_results('test_own', save_name, save_list)

        ############################## TEST FOR TEST SET #############################
        if self.args.test_own is None:
            for idx_data, d in enumerate(self.loader_test):
                for idx_scale, scale in enumerate(self.scale):
                    d.dataset.set_scale(idx_scale)
                    tot_ssim =0
                    tot_bit =0 
                    tot_psnr =0.0
                    i=0
                    for lr, hr, filename in tqdm(d, ncols=80):
                        i+=1
                        lr, hr = self.prepare(lr, hr)
                        self.set_bit(teacher=True)
                        if self.args.test_patch:
                            sr, feat, bit = self.patch_inference(self.model, lr, idx_scale)
                            img_bit = bit
                        else:
                            with torch.no_grad():
                                sr, feat, bit = self.model(lr, idx_scale)
                            img_bit = bit.mean() / self.num_quant_modules
                        self.set_bit(teacher=False)

                        sr = utility.quantize(sr, self.args.rgb_range)
                        save_list = [sr]
                        psnr, ssim = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)

                        tot_bit += img_bit
                        tot_psnr += psnr
                        tot_ssim += ssim

                        if self.args.save_gt:
                            save_list.extend([lr, hr])

                        if self.args.save_results:
                            save_name = '{}_x{}_{:.2f}dB'.format(filename[0], scale, cur_psnr)
                            self.ckp.save_results(d, save_name, save_list)

                    tot_psnr /= len(d)
                    tot_ssim /= len(d)
                    tot_bit /= len(d)

                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} \t SSIM: {:.4f} \tBit: {:.2f}'.format(
                            d.dataset.name,
                            scale,
                            tot_psnr,
                            tot_ssim,
                            tot_bit.item(),
                        )
                    )

        if self.args.save_results:
            self.ckp.end_background()

        torch.set_grad_enabled(True)



    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            # return self.epoch >= self.args.epochs
            return self.epoch > self.args.epochs
