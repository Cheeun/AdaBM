import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

class test2k(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(test2k, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'DIV2K', self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR')
        self.ext = ('', '.png')
