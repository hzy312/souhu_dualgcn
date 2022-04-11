#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：souhu_dualgcn 
@File ：train_parse.py
@Author ：Huang ZiYang
@Date ：2022/4/8 21:19
"""
import json
import spacy
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
import pickle

if __name__ == '__main__':
    # uncomment this to enforce to use gpu
    # spacy.require_gpu()
    nlp = spacy.load("zh_core_web_trf", disable=["tagger", "attribute_ruler", "ner"])
    plmpath = '../roberta-zh'
    tokenizer = AutoTokenizer.from_pretrained(plmpath)
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    pad_id = tokenizer.pad_token_id
    test_file = 'test.txt'
    max_num_tokens = 512
    processed_results = []
    with open(test_file, 'r') as f:
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
            entity = instance['entity']
            context_encoding = tokenizer.encode_plus(tokens, add_special_tokens=False, is_split_into_words=True)
            word_ids = context_encoding.word_ids()
            context_tokens = context_encoding['input_ids']
            len_context_tokens = len(context_tokens)
            for e in entity:
                e_tokens = tokenizer.encode(e, add_special_tokens=False)
                if 1 + len_context_tokens + 1 + len(e_tokens) + 1 > max_num_tokens:
                    temp_matrix = torch.zeros([max_num_tokens, max_num_tokens])
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
                    for j in range(1, len_context_tokens - residual + 1):
                        for k in range(1, len_context_tokens - residual + 1):
                            temp_matrix[j][k] = dep_rel_matrix[word_ids[left_res + j - 1]][word_ids[left_res + k - 1]]
                    temp_matrix *= torch.ones([max_num_tokens, max_num_tokens]) - torch.diag(torch.ones(max_num_tokens))
                    temp_matrix = temp_matrix + temp_matrix.T
                    temp_matrix = temp_matrix + torch.diag(torch.ones(max_num_tokens))
                    for j in range(max_num_tokens - 1 - len(e_tokens), max_num_tokens - 1):
                        for k in range(max_num_tokens):
                            temp_matrix[j][k] = 1
                    entity_mask = torch.zeros(max_num_tokens)
                    entity_mask[max_num_tokens - 1 - len(e_tokens):max_num_tokens - 1] = 1
                    input_encoding = {'tokens': torch.LongTensor(input_tokens),
                                      'dep_rel_matrix': torch.FloatTensor(temp_matrix),
                                      'entity_mask': entity_mask,
                                      'attention_mask': torch.ones(max_num_tokens),
                                      'segment_ids': torch.LongTensor(segment_ids),
                                      'src_mask': src_mask,
                                      'entity': e,
                                      'id': idx}
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
                    for j in range(1, 1 + len_context_tokens):
                        for k in range(1, 1 + len_context_tokens):
                            temp_matrix[j][k] = dep_rel_matrix[word_ids[j - 1]][word_ids[k - 1]]
                    temp_matrix *= torch.ones([max_num_tokens, max_num_tokens]) - torch.diag(torch.ones(max_num_tokens))
                    temp_matrix = temp_matrix + temp_matrix.T
                    temp_matrix = temp_matrix + torch.diag(torch.ones(max_num_tokens))
                    for j in range(2 + len_context_tokens, 2 + len_context_tokens + len(e_tokens)):
                        for k in range(max_num_tokens):
                            temp_matrix[j][k] = 1
                    entity_mask = torch.zeros(max_num_tokens)
                    entity_mask[2 + len_context_tokens:2 + len_context_tokens + len(e_tokens)] = 1
                    input_encoding = {'tokens': torch.LongTensor(input_tokens),
                                      'dep_rel_matrix': torch.FloatTensor(temp_matrix),
                                      'entity_mask': entity_mask,
                                      'attention_mask': attention_mask,
                                      'segment_ids': torch.LongTensor(segment_ids),
                                      'src_mask': src_mask,
                                      'entity': e,
                                      'id': idx}

                processed_results.append(input_encoding)

    with open('test_processed.txt', 'wb') as f:
        pickle.dump(processed_results, f)

    print('{} test instances!'.format(len(processed_results)))
    with open('length.txt', 'w') as f:
        f.write('{} test instances!'.format(len(processed_results)))
