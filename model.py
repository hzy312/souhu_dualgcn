#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：souhu_dualgcn 
@File ：model.py
@Author ：Huang ZiYang
@Date ：2022/4/10 22:09 
"""

from torch import nn
import torch
import torch.nn.functional as F
import copy
import math


cols = ['tokens', 'dep_rel_matrix', 'entity_mask', 'attention_mask', 'segment_ids', 'src_mask']

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class DualGCNBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        self.classifier = nn.Linear(opt.bert_dim * 2, opt.polarities_dim)

    def forward(self, inputs):
        outputs1, outputs2, adj_ag, adj_dep, pooled_output = self.gcn_model(inputs)
        # bert_dim/2 + bert_dim/2 + bert_dim
        # syn_embedding + sem_embedding + all_embedding
        final_outputs = torch.cat((outputs1, outputs2, pooled_output), dim=-1)
        logits = self.classifier(final_outputs)

        # [bsz, l, l]
        adj_ag_T = adj_ag.transpose(1, 2)
        # [l, l]
        identity = torch.eye(adj_ag.size(1))
        # [bsz, l, l]
        identity = identity.unsqueeze(0).expand(adj_ag.size(0), adj_ag.size(1), adj_ag.size(1))
        # orthogonal regularization [bsz, l, l]
        ortho = adj_ag @ adj_ag_T

        for i in range(ortho.size(0)):
            ortho[i] -= torch.diag(torch.diag(ortho[i]))
            ortho[i] += torch.eye(ortho[i].size(0))

        penal = None
        if self.opt.losstype == 'doubleloss':
            penal1 = (torch.norm(ortho - identity) / adj_ag.size(0))
            penal2 = (adj_ag.size(0) / torch.norm(adj_ag - adj_dep))
            penal = self.opt.alpha * penal1 + self.opt.beta * penal2

        elif self.opt.losstype == 'orthogonalloss':
            penal = (torch.norm(ortho - identity) / adj_ag.size(0))
            penal = self.opt.alpha * penal

        elif self.opt.losstype == 'differentiatedloss':
            penal = (adj_ag.size(0) / torch.norm(adj_ag - adj_dep))
            penal = self.opt.beta * penal

        return logits, penal


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers)

    def forward(self, inputs):
        inputs = [inputs[c] for c in cols]
        text_bert_indices, adj_dep, aspect_mask, attention_mask, bert_segments_ids, src_mask = inputs
        h1, h2, adj_ag, pooled_output = self.gcn(adj_dep, inputs)

        # avg pooling asp feature
        # [bsz, l] ---> [bsz, 1]
        # count the num of aspect tokens for every instance
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        # [bsz, l, h]
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim // 2)
        # [bsz, h]
        outputs1 = (h1 * aspect_mask).sum(dim=1) / asp_wn
        outputs2 = (h2 * aspect_mask).sum(dim=1) / asp_wn
        return outputs1, outputs2, adj_ag, adj_dep, pooled_output


class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        # bertmodel
        self.bert = bert
        # options
        self.opt = opt
        # num of gcn layers: default 2
        self.layers = num_layers
        # 768//2 = 384
        self.mem_dim = opt.bert_dim // 2
        # 1
        self.attention_heads = opt.attention_heads
        # 768
        self.bert_dim = opt.bert_dim
        # 0.3
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        # 0.3
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        # 0.1
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        # ..
        self.layernorm = LayerNorm(opt.bert_dim)

        # gcn layer
        self.W = nn.ModuleList()
        # 2 layer
        # 768 --> 384 --> 384
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        # self attention
        self.attn = MultiHeadAttention(opt.attention_heads, self.bert_dim)
        self.weight_list = nn.ModuleList()
        # 768 ---> 384 ---> 384
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        # 384 * 384
        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        # 384 * 384
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

    def forward(self, adj, inputs):
        text_bert_indices, adj_dep, aspect_mask, attention_mask, bert_segments_ids, src_mask = inputs
        # [bsz, -1, l]
        src_mask = src_mask.unsqueeze(-2)

        # seq_output: [bsz, l, hidden]
        # pooled_output: [bsz, hidden] cls embedding
        outputs = self.bert(text_bert_indices, attention_mask=attention_mask,
                                                   token_type_ids=bert_segments_ids)
        sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output
        # layernorm + dropout
        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)
        pooled_output = self.pooled_drop(pooled_output)

        # adj: [bsz, l, l] ---> [bsz, l] ---> [bsz, l, 1]
        denom_dep = adj.sum(2).unsqueeze(2) + 1
        # attention score:[bsz, num_heads, l, l]
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        # squeeze() [bsz, 1, l, l] --> [bsz, l, l]
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        multi_head_list = []
        outputs_dep = None
        adj_ag = None

        # * Average Multi-head Attention matrixes
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag += attn_adj_list[i]
        # [bsz, l, l]
        adj_ag = adj_ag / self.attention_heads

        # fill 1 in the diagonal
        for j in range(adj_ag.size(0)):
            adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
            adj_ag[j] += torch.eye(adj_ag[j].size(0))
        # [bsz, l, l]
        adj_ag = src_mask.transpose(1, 2) * adj_ag

        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        # sem and syn initialize with the bert output
        outputs_ag = gcn_inputs
        outputs_dep = gcn_inputs

        for l in range(self.layers):
            # ************SynGCN*************
            # [bsz, l, l] x [bsz, l, hidden] ---> [bsz, l, hidden]
            Ax_dep = adj.bmm(outputs_dep)
            AxW_dep = self.W[l](Ax_dep)
            AxW_dep = AxW_dep / denom_dep
            gAxW_dep = F.relu(AxW_dep)

            # ************SemGCN*************
            # [bsz, l, hidden]
            Ax_ag = adj_ag.bmm(outputs_ag)
            AxW_ag = self.weight_list[l](Ax_ag)
            AxW_ag = AxW_ag / denom_ag
            gAxW_ag = F.relu(AxW_ag)

            # * mutual Biaffine module
            # [bsz, l, h] x [h, h] x [bsz, h, l] ---> [bsz, l, l]
            A1 = F.softmax(torch.bmm(torch.matmul(gAxW_dep, self.affine1), torch.transpose(gAxW_ag, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine2), torch.transpose(gAxW_dep, 1, 2)), dim=-1)
            # [bsz, l, l] x [bsz, l, h] ---> [bsz, l, h]
            gAxW_dep, gAxW_ag = torch.bmm(A1, gAxW_ag), torch.bmm(A2, gAxW_dep)
            outputs_dep = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep
            outputs_ag = self.gcn_drop(gAxW_ag) if l < self.layers - 1 else gAxW_ag

        return outputs_ag, outputs_dep, adj_ag, pooled_output


def attention(query, key, mask=None, dropout=None):
    # assumption: the shape of query is [bsz, l, h]
    d_k = query.size(-1)
    # [bsz, l, h] [bsz, h, l] ---> [bsz, l, l]  ij--->(qi, k_j)'s attention score
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # mask the padding
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# did this copied from the pytorch MHSA?
class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        # 1, 768, 0.1
        # scalable, but not used in this code
        super(MultiHeadAttention, self).__init__()
        # h: num_heads
        # d_model: hidden_size
        # 768 // 12 == 64
        assert d_model % h == 0
        # head_side
        self.d_k = d_model // h
        self.h = h
        # return a ModuleList of x copies of the origin module
        # q_projection matrix, k_projection matrix
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        # query: [bsz, l, h]
        # mask: [bsz, l]
        mask = mask[:, :, :query.size(1)]
        # mask [b, l, l]
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        # [bsz, l, num_heads, head_size] ---> [bsz, num_heads, l, head_size]
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn

