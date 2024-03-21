import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

class div2k_valid(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(div2k_valid, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'DIV2K' )
        self.dir_hr = os.path.join(self.apath, 'DIV2K_valid_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_valid_LR_bicubic')
        self.ext = ('', '.png')
