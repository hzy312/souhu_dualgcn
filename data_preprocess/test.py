#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：souhu_dualgcn 
@File ：test.py
@Author ：Huang ZiYang
@Date ：2022/4/10 21:13 
"""
import pickle
processed_file = 'example_processed.txt'
with open(processed_file, 'rb') as f:
    data = pickle.load(f)

print(data)