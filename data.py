#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：souhu_dualgcn 
@File ：data.py
@Author ：Huang ZiYang
@Date ：2022/4/8 21:10 
"""
import pickle

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from argparse import Namespace


class SOUHUDataset(Dataset):
    def __init__(self, data):
        super(SOUHUDataset, self).__init__()
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class SOUHUDataModule(pl.LightningDataModule):
    def __init__(self, opt: Namespace):
        super(SOUHUDataModule, self).__init__()
        self.opt = opt
        self.save_hyperparameters()
        with open(opt.train_data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.train_ratio = opt.train_ratio
        self.train_size = int(len(self.data) * self.train_ratio)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_data = self.data[:self.train_size]
        dataloader = DataLoader(train_data,
                                batch_size=self.opt.batch_size,
                                shuffle=False,
                                num_workers=4)
        return dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_data = self.data[self.train_size:]
        dataloader = DataLoader(val_data,
                                batch_size=self.opt.batch_size,
                                num_workers=4)
        return dataloader
