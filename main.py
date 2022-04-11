#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：souhu_dualgcn 
@File ：main.py
@Author ：Huang ZiYang
@Date ：2022/4/11 16:25 
"""
from option import get_parser
import os
import json
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
import pytorch_lightning as pl
from trainer import SOUHUModule


def main():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    lr_callback = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_path,
        filename='{epoch:02d}-{val_f1:.5f}',
        save_top_k=args.save_topk,
        save_last=True,
        monitor="val_f1",
        mode="max",
    )

    # save args
    with open(os.path.join(args.save_path, "args.json"), 'w') as f:
        args_dict = args.__dict__
        json.dump(args_dict, f, indent=4)

    pl.seed_everything(42)
    trainer = Trainer.from_argparse_args(args,
                                         callbacks=[checkpoint_callback,
                                                    ModelSummary(max_depth=-1),
                                                    lr_callback],
                                         deterministic=True)
    model = SOUHUModule(args)
    torch.cuda.empty_cache()
    trainer.fit(model)


if __name__ == '__main__':
    main()
