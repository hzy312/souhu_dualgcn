#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：souhu_dualgcn 
@File ：predict.py
@Author ：Huang ZiYang
@Date ：2022/4/11 21:09 
"""
from option import get_parser
import torch
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pickle
from trainer import SOUHUModule
from collections import defaultdict
import json


class TestDataset(Dataset):
    def __init__(self, path):
        super(TestDataset, self).__init__()
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def main():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    pl.seed_everything(42)
    model = SOUHUModule.load_from_checkpoint(args.predict_ckpt_path)
    model.eval()
    trainer = Trainer()
    torch.cuda.empty_cache()
    test_dataset = TestDataset(args.predict_data_path)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=5)
    results = trainer.predict(model, test_dataloader)
    predictions = []
    res_dict = defaultdict(dict)
    for temp in results:
        predictions += temp
    for temp in predictions:
        res_dict[temp[0]][temp[1]] = temp[2]

    with open(args.result_path, 'w') as f:
        f.write('id\tresult\n')
        for idx, r in res_dict.items():
            f.write(str(idx)+'\t')
            f.write(json.dumps(r, ensure_ascii=False)+'\n')









    print(predictions)


if __name__ == '__main__':
    main()
