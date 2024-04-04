import torch
import torch.nn as nn
import utility
import data
import model
from option import args
from trainer import Trainer

import time
import datetime

import numpy
import random

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
numpy.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    exp_start_time = time.time()
    _loader = data.Data(args)
    _model = model.Model(args, checkpoint)
    t = Trainer(args, _loader, _model, checkpoint)
    
    # t.test_teacher()
    while not t.terminate():
        torch.manual_seed(args.seed)
        t.train()   
        t.test()

    exp_end_time = time.time()
    exp_time_interval = exp_end_time - exp_start_time
    t_string = "Total Running Time is: " + str(datetime.timedelta(seconds=exp_time_interval)) + "\n"
    checkpoint.write_log('{}'.format(t_string))
    checkpoint.done()