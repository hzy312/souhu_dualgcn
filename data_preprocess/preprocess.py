#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：souhu_dualgcn 
@File ：preprocess.py
@Author ：Huang ZiYang
@Date ：2022/4/12 12:54 
"""
import json
import spacy
from tqdm import tqdm
import torch
import pickle
import argparse

if __name__ == '__main__':
    # uncomment this to enforce to use gpu
    # spacy.require_gpu()
    nlp = spacy.load("zh_core_web_trf", disable=["tagger", "attribute_ruler", "ner"])
    plmpath = '../roberta-zh'
    train_file = 'train.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='the input file')
    parser.add_argument('--output_file', type=str, required=True, help='the output file')
    args = parser.parse_args()
    max_num_tokens = 512
    processed_results = []
    with open(args.input_file, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            instance = json.loads(line)
            content = instance['content']
            idx = instance['id']
            doc = nlp(content)
            dep_rel_matrix = torch.zeros([len(doc), len(doc)])
            char2token = {}
            tokens = []
            for i, token in enumerate(doc):
                tokens.append(token.text)
                char2token[token.idx] = i
            for token in doc:
                dep_rel_matrix[char2token[token.idx], char2token[token.head.idx]] = 1
            if 'train' in args.input_file:
                temp_ins = {'tokens': tokens, 'dep_rel_matrix': dep_rel_matrix, 'entity': instance['entity']}
            else:
                temp_ins = {'idx': instance['id'], 'tokens': tokens, 'dep_rel_matrix': dep_rel_matrix,
                            'entity': instance['entity']}
            processed_results.append(temp_ins)

    print('{} instances in {} !'.format(len(processed_results), args.input_file))
    with open('len_' + args.input_file, 'w') as f:
        f.write('{} instances in {} !'.format(len(processed_results), args.input_file))
    with open(args.output_file, 'wb') as f:
        pickle.dump(processed_results, f)
