#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：souhu_dualgcn 
@File ：eda.py
@Author ：Huang ZiYang
@Date ：2022/4/8 23:16 
"""
import json
from tqdm import tqdm
from transformers import AutoTokenizer

# 有他娘的22506条数据都超过512了

# res = []
# tokenizer = AutoTokenizer.from_pretrained('../roberta-zh')
# with open('train.txt', 'r') as f:
#     lines = f.readlines()
#     for line in tqdm(lines):
#         temp = json.loads(line)
#         content = temp['content']
#         entity = temp['entity']
#         idx = temp['id']
#         for e in entity.keys():
#             l = len(tokenizer.encode(content, e))
#             if l > 512:
#                 t = {'id':idx, 'len':l, 'entity':e, 'context':content}
#                 res.append(t)
#
# with open('len_statistics.json', 'w') as f:
#     json.dump(res, f, indent=2)


# with open('len_statistics.json', 'r') as f:
#     data = json.load(f)
#     print(data)