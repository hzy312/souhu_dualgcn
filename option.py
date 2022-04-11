#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：souhu_dualgcn 
@File ：option.py
@Author ：Huang ZiYang
@Date ：2022/4/11 16:25 
"""
import argparse


def get_parser():
    parser = argparse.ArgumentParser('hyperparameters')
    parser.add_argument('--bert_dim',
                        type=int, default=768)
    parser.add_argument('--polarities_dim',
                        default=5, type=int, help='5')
    parser.add_argument('--losstype',
                        default='doubleloss', type=str,
                        help="['doubleloss', 'orthogonalloss', 'differentiatedloss']")
    parser.add_argument('--alpha',
                        default=0.25, type=float)
    parser.add_argument('--beta',
                        default=0.25, type=float)
    parser.add_argument('--num_layers',
                        type=int, default=2, help='Num of GCN layers.')
    parser.add_argument('--attention_heads',
                        default=1, type=int, help='number of multi-attention heads')
    parser.add_argument('--bert_dropout',
                        default=0.3, help='dropout prob for bert outputs')
    parser.add_argument('--weight_decay',
                        type=float, default=1e-5, help='weight decay if we apply some')
    parser.add_argument('--gcn_dropout',
                        type=float, default=0.1, help='dropout prob for gcn layers')
    parser.add_argument('--warmup_steps',
                        type=int, default=0, help='warmup steps if we apply')
    parser.add_argument('--batch_size',
                        default=32, type=int, help='batch_size')
    parser.add_argument('--train_data_path',
                        type=str, default='./data_preprocess/train_processed.txt', help='training data path')
    parser.add_argument('--predict_data_path',
                        type=str, default='./data_preprocess/test_processed.txt', help='predict data path')
    parser.add_argument('--predict_ckpt_path',
                        type=str, default='./save/last.ckpt', help='the best model to predict')
    parser.add_argument('--plm_path',
                        type=str, default='./roberta-zh', help='pretrained model path')
    parser.add_argument('--save_path',
                        type=str, default='./save', help='ckpt save path')
    parser.add_argument('--save_topk',
                        type=int, default=3)
    parser.add_argument('--result_path',
                        type=str, default='prediction.txt', help='test result save path')
    parser.add_argument('--train_ratio',
                        type=float, default=0.98, help='train val split')
    parser.add_argument('--lr',
                        type=float, default=1e-3, help='initial learning rate')

    return parser
