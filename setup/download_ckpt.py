#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：souhu_dualgcn 
@File ：download_ckpt.py
@Author ：Huang ZiYang
@Date ：2022/4/11 19:41 
"""
from transformers import AutoModel
plm_name = 'hfl/chinese-roberta-wwm-ext-large'
path = '../roberta-zh'
model = AutoModel.from_pretrained(plm_name)
model.save_pretrained(path)
