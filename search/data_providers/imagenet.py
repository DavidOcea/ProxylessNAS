# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import torch.utils.data

from data_providers.base_provider import *
from .datasets import FileListLabeledDataset,GivenSizeSampler
from .transforms import RandomResizedCrop, Compose, Resize, CenterCrop, ToTensor, \
	Normalize, RandomHorizontalFlip, ColorJitter, Lighting

import numpy as np

class ImagenetDataProvider(DataProvider):
    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=32, resize_scale=0.08, distort_color=None):

        self._save_path = save_path
        num_tasks = len(self.train_path)
        train_transforms = self.build_train_transform(distort_color, resize_scale)
        self.train_dataset = FileListLabeledDataset(self.train_list[0], self.train_path[0], train_transforms)

        if valid_size is not None:
            self.valid_dataset = FileListLabeledDataset(self.valid_list[0], self.valid_path[0], Compose([
                Resize(self.resize_value),
                CenterCrop(self.image_size),
                ToTensor(),
                self.normalize,
            ])) 

            train_sampler = GivenSizeSampler(self.train_dataset, total_size = len(self.train_dataset), rand_seed = 0)
            # import pdb; pdb.set_trace()
            self.train = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=False,
            ) 
            self.valid = torch.utils.data.DataLoader(
                self.valid_dataset, batch_size=valid_size,
                num_workers=n_worker, pin_memory=False,
            ) 
        else:
            self.train = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=False,
            ) 
            self.valid = None

        self.test_dataset = FileListLabeledDataset(self.valid_list[0], self.valid_path[0], Compose([
                Resize(self.resize_value),
                CenterCrop(self.image_size),
                ToTensor(),
                self.normalize,
            ]))

        self.test = torch.utils.data.DataLoader(
            self.test_dataset[0], batch_size=test_batch_size, shuffle=False, num_workers=n_worker, pin_memory=False,
        ) 

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'wm_data'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 1000

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = ['/workspace/fanhao/bk_cls8/data/cls8_v0.2.02'] #WM_data_20191113
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download ImageNet')

    @property
    def train_path(self):
        return self.save_path

    @property
    def train_list(self):
        return ['/workspace/yangdecheng/code/ProxylessNAS/search/data/bk_train_nas.txt'] 

    @property
    def valid_path(self):
        return self._save_path
    
    @property
    def valid_list(self):
        return ['/workspace/yangdecheng/code/ProxylessNAS/search/data/bk_val.txt']

    @property
    def normalize(self):
        return Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])

    def build_train_transform(self, distort_color, resize_scale):
        print('Color jitter: %s' % distort_color)
        if distort_color == 'strong':
            color_transform = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif distort_color == 'normal':
            color_transform = ColorJitter(brightness=[0.5,1.5], contrast=[0.5,1.5], saturation=[0.5,1.5], hue= 0) #brightness=32. / 255., saturation=0.5
        else:
            color_transform = None
        if color_transform is None:
            train_transforms = Compose([
                RandomResizedCrop(self.image_size, scale=(resize_scale, 1.0)),
                RandomHorizontalFlip(),
                ToTensor(),
                self.normalize,
            ])
        else:
            train_transforms = Compose([
                RandomResizedCrop(self.image_size, scale=(resize_scale, 1.0)),
                RandomHorizontalFlip(),
                color_transform,
                ToTensor(),
                self.normalize,
            ])
        return train_transforms

    @property
    def resize_value(self):
        return [256,256] #256

    @property
    def image_size(self):
        return 224 #224


