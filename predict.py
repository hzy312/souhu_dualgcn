#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：souhu_dualgcn 
@File ：predict.py
@Author ：Huang ZiYang
@Date ：2022/4/11 21:09 
"""
from tqdm import  tqdm
from option import get_parser
import torch
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pickle
from trainer import SOUHUModule
from collections import defaultdict
import json
from transformers import AutoTokenizer


class TestDataset(Dataset):
    def __init__(self, path, tokenizer):
        super(TestDataset, self).__init__()
        self.encoding = []
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        max_num_tokens = 512
        sep_id = tokenizer.sep_token_id
        cls_id = tokenizer.cls_token_id
        pad_id = tokenizer.pad_token_id
        for instance in tqdm(self.data):
            entity = instance['entity']
            tokens = instance['tokens']
            dep_rel_matrix = instance['dep_rel_matrix']
            idx = instance['idx']
            context_encoding = tokenizer.encode_plus(tokens, add_special_tokens=False, is_split_into_words=True)
            word_ids = torch.LongTensor(context_encoding.word_ids())
            context_tokens = context_encoding['input_ids']
            len_context_tokens = len(context_tokens)
            for e in entity:
                e_tokens = tokenizer.encode(e, add_special_tokens=False)
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
                    row, col = torch.arange(1, 1 + len_context_tokens), torch.arange(1, 1 + len_context_tokens)
                    row_ids, col_ids = word_ids[row - 1], word_ids[col - 1]
                    car_prod = torch.cartesian_prod(row_ids, col_ids)
                    temp_matrix[1:1 + len_context_tokens, 1:1 + len_context_tokens] = dep_rel_matrix[
                        car_prod[:, 0], car_prod[:, 1]].reshape(len_context_tokens, len_context_tokens)
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
                self.encoding.append(input_encoding)
        print('{} pred instances!'.format(len(self.encoding)))


    def __getitem__(self, item):
        return self.encoding[item]

    def __len__(self):
        return len(self.encoding)


def main():
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    pl.seed_everything(42)
    model = SOUHUModule.load_from_checkpoint(args.predict_ckpt_path, opt=args)
    model.eval()
    model.freeze()
    trainer = Trainer().from_argparse_args(args)
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
    test_dataset = TestDataset(args.predict_data_path, tokenizer)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=10)
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




if __name__ == '__main__':
    main()
