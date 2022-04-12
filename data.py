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
import torch
from transformers import AutoTokenizer
from tqdm import tqdm


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
        self.tokenizer = AutoTokenizer.from_pretrained(self.opt.plm_path)
        self.encoding = []
        max_num_tokens = 512
        sep_id = self.tokenizer.sep_token_id
        cls_id = self.tokenizer.cls_token_id
        pad_id = self.tokenizer.pad_token_id
        for instance in tqdm(self.data):
            tokens = instance['tokens']
            entity = instance['entity']
            dep_rel_matrix = instance['dep_rel_matrix']
            context_encoding = self.tokenizer.encode_plus(tokens, add_special_tokens=False, is_split_into_words=True)
            word_ids = torch.LongTensor(context_encoding.word_ids())
            context_tokens = context_encoding['input_ids']
            len_context_tokens = len(context_tokens)
            for e, l in entity.items():
                e_tokens = self.tokenizer.encode(e, add_special_tokens=False)
                if 1 + len_context_tokens + 1 + len(e_tokens) + 1 > max_num_tokens:
                    residual = len_context_tokens + len(e_tokens) + 3 - max_num_tokens
                    # cut the left and right
                    if residual % 2 == 0:
                        left_res = residual // 2
                        right_res = left_res
                    else:
                        left_res = residual // 2
                        right_res = left_res + 1
                    input_tokens = [cls_id] + context_tokens[left_res: len_context_tokens - right_res] + [
                        sep_id] + e_tokens + [sep_id]
                    assert len(input_tokens) == max_num_tokens, 'wrong truncation!'
                    segment_ids = [0] * (1 + len_context_tokens - residual + 1) + [1] * (len(e_tokens) + 1)
                    assert len(segment_ids) == max_num_tokens
                    src_mask = torch.ones(max_num_tokens)
                    src_mask[0] = 0
                    src_mask[1 + len_context_tokens - residual] = 0
                    src_mask[-1] = 0
                    temp_matrix = torch.zeros([max_num_tokens, max_num_tokens])
                    row, col = torch.arange(1, len_context_tokens - residual + 1), torch.arange(1, len_context_tokens - residual + 1)
                    row_ids, col_ids = word_ids[left_res+row-1], word_ids[left_res+col-1]
                    car_prod = torch.cartesian_prod(row_ids, col_ids)
                    temp_matrix[1:1 + len_context_tokens-residual, 1:1 + len_context_tokens - residual] = dep_rel_matrix[
                        car_prod[:, 0], car_prod[:, 1]].reshape(len_context_tokens-residual, len_context_tokens-residual)
                    temp_matrix *= torch.ones([max_num_tokens, max_num_tokens]) - torch.diag(torch.ones(max_num_tokens))
                    temp_matrix = temp_matrix + temp_matrix.T
                    temp_matrix = temp_matrix + torch.diag(torch.ones(max_num_tokens))
                    for j in range(max_num_tokens - 1 - len(e_tokens), max_num_tokens - 1):
                        for k in range(max_num_tokens):
                            temp_matrix[j][k] = 1
                    entity_mask = torch.zeros(max_num_tokens)
                    entity_mask[max_num_tokens - 1 - len(e_tokens):max_num_tokens - 1] = 1
                    # label: [-2, -1, 0, 1, 2] ---> [0, 1, 2, 3, 4]
                    input_encoding = {'tokens': torch.LongTensor(input_tokens),
                                      'dep_rel_matrix': torch.FloatTensor(temp_matrix),
                                      'label': int(l) + 2,
                                      'entity_mask': entity_mask,
                                      'attention_mask': torch.ones(max_num_tokens),
                                      'segment_ids': torch.LongTensor(segment_ids),
                                      'src_mask': src_mask}
                else:
                    input_tokens = [cls_id] + context_tokens + [sep_id] + e_tokens + [sep_id]
                    attention_mask = torch.ones(max_num_tokens)
                    len_useful = len(input_tokens)
                    attention_mask[len_useful:] = 0
                    input_tokens = input_tokens + [pad_id] * (max_num_tokens - len_useful)
                    assert len(input_tokens) == 512, 'wrong padding'
                    segment_ids = [0] * (1 + len_context_tokens + 1) + [1] * (len(e_tokens) + 1) + [0] * (
                            max_num_tokens - len_useful)
                    assert len(segment_ids) == max_num_tokens
                    src_mask = torch.ones(max_num_tokens)
                    src_mask[0] = 0
                    src_mask[1 + len_context_tokens] = 0
                    src_mask[-1] = 0
                    temp_matrix = torch.zeros([max_num_tokens, max_num_tokens])
                    row, col = torch.arange(1, 1 + len_context_tokens), torch.arange(1, 1 + len_context_tokens)
                    row_ids, col_ids = word_ids[row - 1], word_ids[col - 1]
                    car_prod = torch.cartesian_prod(row_ids, col_ids)
                    temp_matrix[1:1 + len_context_tokens, 1:1 + len_context_tokens] = dep_rel_matrix[
                        car_prod[:, 0], car_prod[:, 1]].reshape(len_context_tokens, len_context_tokens)
                    temp_matrix *= torch.ones([max_num_tokens, max_num_tokens]) - torch.diag(torch.ones(max_num_tokens))
                    temp_matrix = temp_matrix + temp_matrix.T
                    temp_matrix = temp_matrix + torch.diag(torch.ones(max_num_tokens))
                    for j in range(2 + len_context_tokens, 2 + len_context_tokens + len(e_tokens)):
                        temp_matrix[j][:] = 1
                    entity_mask = torch.zeros(max_num_tokens)
                    entity_mask[2 + len_context_tokens:2 + len_context_tokens + len(e_tokens)] = 1
                    input_encoding = {'tokens': torch.LongTensor(input_tokens),
                                      'dep_rel_matrix': torch.FloatTensor(temp_matrix),
                                      'label': int(l) + 2,
                                      'entity_mask': entity_mask,
                                      'attention_mask': attention_mask,
                                      'segment_ids': torch.LongTensor(segment_ids),
                                      'src_mask': src_mask}
                self.encoding.append(input_encoding)
        print('{} training instances!'.format(len(self.encoding)))
        self.train_ratio = opt.train_ratio
        self.train_size = int(len(self.encoding) * self.train_ratio)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_data = self.encoding[:self.train_size]
        dataset = SOUHUDataset(train_data)
        dataloader = DataLoader(dataset,
                                batch_size=self.opt.batch_size,
                                shuffle=False,
                                num_workers=10)
        return dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_data = self.encoding[self.train_size:]
        dataset = SOUHUDataset(val_data)
        dataloader = DataLoader(dataset,
                                batch_size=self.opt.batch_size,
                                num_workers=10)
        return dataloader
