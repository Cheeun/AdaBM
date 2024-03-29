from importlib import import_module
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

import random
import torch.utils.data as data

import numpy
import torch

class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=d))

            train_dataset = MyConcatDataset(datasets)
            indices = random.sample(range(0, len(train_dataset)), args.num_data) 
            # sampled over (train_dataset[0]=001.png~train_dataset[799]=800.png)

            print('Indices of {} sampled data...'.format(len(indices)))
            print(indices)

            train_dataset_sampled = data.Subset(train_dataset, indices)

            self.loader_train = dataloader.DataLoader(
                train_dataset_sampled,
                batch_size=args.batch_size_update,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )

        self.loader_init = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=d))

            self.loader_init = dataloader.DataLoader(
                train_dataset_sampled,
                batch_size=args.batch_size_calib,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )
        
  




