#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/6 17:37
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : attention_layers.py
"""

import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_

from functools import partial
from Models.configs import *


def attention(query, key, value,
              mask=None, dropout=None, weight=None,
              attention_type='softmax'):
    '''
    Simplified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    Compute the Scaled Dot Product Attention
    '''

    d_k = query.size(-1)

    if attention_type == 'cosine':
        p_attn = F.cosine_similarity(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
    else:
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        seq_len = scores.size(-1)

        if attention_type == 'softmax':
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, dim=-1)
        elif attention_type in ['fourier', 'integral', 'local']:
            if mask is not None:
                scores = scores.masked_fill(mask == 0, 0)
            p_attn = scores / seq_len

    if dropout is not None:
        p_attn = F.dropout(p_attn)

    out = torch.matmul(p_attn, value)

    return out, p_attn


def linear_attention(query, key, value,
                     mask=None, dropout=None,
                     attention_type='galerkin'):
    '''
    Adapted from lucidrains' implementaion
    https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    to https://nlp.seas.harvard.edu/2018/04/03/attention.html template
    linear_attn function
    Compute the Scaled Dot Product Attention globally
    '''

    seq_len = query.size(-2)
    if attention_type in ['linear', 'global']:
        query = query.softmax(dim=-1)
        key = key.softmax(dim=-2)
    scores = torch.matmul(key.transpose(-2, -1), value)

    if mask is not None:
        raise RuntimeError("linear attention does not support casual mask.")

    p_attn = scores / seq_len

    if dropout is not None:
        p_attn = F.dropout(p_attn)

    out = torch.matmul(query, p_attn)
    return out, p_attn


def causal_linear_attn(query, key, value, kv_mask=None, dropout=None, eps=1e-7):
    '''
    Modified from https://github.com/lucidrains/linear-attention-transformer
    '''
    bsz, n_head, seq_len, d_k, dtype = *query.shape, query.dtype

    key /= seq_len

    if kv_mask is not None:
        mask = kv_mask[:, None, :, None]
        key = key.masked_fill_(~mask, 0.)
        value = value.masked_fill_(~mask, 0.)
        del mask

    b_q, b_k, b_v = [x.reshape(bsz, n_head, -1, 1, d_k) for x in (query, key, value)]

    b_k_sum = b_k.sum(dim=-2)
    b_k_cumsum = b_k_sum.cumsum(dim=-2).type(dtype)

    p_attn = torch.einsum('bhund,bhune->bhude', b_k, b_v)
    p_attn = p_attn.cumsum(dim=-3).type(dtype)
    if dropout is not None:
        p_attn = F.dropout(p_attn)

    D_inv = 1. / torch.einsum('bhud,bhund->bhun', b_k_cumsum + eps, b_q)
    attn = torch.einsum('bhund,bhude,bhun->bhune', b_q, p_attn, D_inv)
    return attn.reshape(*query.shape), p_attn


class PositionalEncoding(nn.Module):
    '''
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    This is not necessary if spacial coords are given
    input is (batch, seq_len, d_model)
    '''

    def __init__(self, d_model,
                 dropout=0.1,
                 max_len=2 ** 13):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(2 ** 13) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """
    FeedForward layer in transformers
    """

    def __init__(self, in_dim=256,
                 dim_feedforward: int = 1024,
                 out_dim=None,
                 batch_norm=False,
                 activation='relu',
                 dropout=0.1):
        super(FeedForward, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        n_hidden = dim_feedforward
        # activation = default(activation, 'relu')
        self.lr1 = nn.Linear(in_dim, n_hidden)
        self.activation = activation_dict[activation]
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(n_hidden)
        self.lr2 = nn.Linear(n_hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch, seq_len, in_dim)
        """
        x = self.activation(self.lr1(x))
        x = self.dropout(x)
        if self.batch_norm:
            x = x.permute((0, 2, 1))
            x = self.bn(x)
            x = x.permute((0, 2, 1))
        x = self.lr2(x)
        return x


class SimpleAttention(nn.Module):
    '''
    The attention is using a vanilla (QK^T)V or Q(K^T V) with no softmax
    For an encoder layer, the tensor size is slighly different from the official pytorch implementation

    attn_types:
        - fourier: integral, local
        - galerkin: global
        - linear: standard linearization
        - softmax: classic softmax attention

    In this implementation, output is (N, L, E).
    batch_first will be added in the next version of PyTorch: https://github.com/pytorch/pytorch/pull/55285

    Reference: code base modified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    - added xavier init gain
    - added layer norm <-> attn norm switch
    - added diagonal init

    In https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    the linear attention in each head is implemented as an Einstein sum
    attn_matrix = torch.einsum('bhnd,bhne->bhde', k, v)
    attn = torch.einsum('bhnd,bhde->bhne', q, attn_matrix)
    return attn.reshape(*q.shape)
    here in our implementation this is achieved by a slower transpose+matmul
    but can conform with the template Harvard NLP gave
    '''

    def __init__(self, n_head, d_model,
                 pos_dim: int = 1,
                 attention_type='fourier',
                 dropout=0.1,
                 xavier_init=1e-4,
                 diagonal_weight=1e-2,
                 symmetric_init=False,
                 norm_add=False,
                 norm_type='layer',
                 eps=1e-5):
        super(SimpleAttention, self).__init__()
        assert d_model % n_head == 0
        self.attention_type = attention_type
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(3)])
        self.xavier_init = xavier_init
        self.diagonal_weight = diagonal_weight
        self.symmetric_init = symmetric_init
        if self.xavier_init > 0:
            self._reset_parameters()
        self.norm_add = norm_add
        self.norm_type = norm_type
        if norm_add:
            self._get_norm(eps=eps)

        if pos_dim > 0:
            self.fc = nn.Linear(d_model + n_head * pos_dim, d_model)

        self.attn_weight = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, pos=None, mask=None, weight=None):
        """
        forward compute
        :param query: (batch, seq_len, d_model)
        :param key: (batch, seq_len, d_model)
        :param value: (batch, seq_len, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        bsz = query.size(0)
        if weight is not None:
            query, key = weight * query, weight * key

        query, key, value = \
            [layer(x).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)
             for layer, x in zip(self.linears, (query, key, value))]

        if self.norm_add:
            if self.attention_type in ['linear', 'galerkin', 'global']:
                if self.norm_type == 'instance':
                    key, value = key.transpose(-2, -1), value.transpose(-2, -1)

                key = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], dim=1)
                value = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_V, (value[:, i, ...] for i in range(self.n_head)))], dim=1)

                if self.norm_type == 'instance':
                    key, value = key.transpose(-2, -1), value.transpose(-2, -1)
            else:
                if self.norm_type == 'instance':
                    key, query = key.transpose(-2, -1), query.transpose(-2, -1)

                key = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], dim=1)
                query = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_Q, (query[:, i, ...] for i in range(self.n_head)))], dim=1)

                if self.norm_type == 'instance':
                    key, query = key.transpose(-2, -1), query.transpose(-2, -1)

        if pos is not None and self.pos_dim > 0:
            assert pos.size(-1) == self.pos_dim
            pos = pos.unsqueeze(1)
            pos = pos.repeat([1, self.n_head, 1, 1])
            query, key, value = [torch.cat([pos, x], dim=-1)
                                 for x in (query, key, value)]

        if self.attention_type in ['linear', 'galerkin', 'global']:
            x, self.attn_weight = linear_attention(query, key, value,
                                                   mask=mask,
                                                   attention_type=self.attention_type,
                                                   dropout=self.dropout)
        elif self.attention_type == 'causal':
            assert mask is not None
            x, self.attn_weight = causal_linear_attn(query, key, value,
                                                     mask=mask,
                                                     dropout=self.dropout)
        else:
            x, self.attn_weight = attention(query, key, value,
                                            mask=mask,
                                            attention_type=self.attention_type,
                                            dropout=self.dropout)

        out_dim = self.n_head * self.d_k if pos is None else self.n_head * \
                                                             (self.d_k + self.pos_dim)
        att_output = x.transpose(1, 2).contiguous().view(bsz, -1, out_dim)

        if pos is not None and self.pos_dim > 0:
            att_output = self.fc(att_output)

        return att_output, self.attn_weight

    def _reset_parameters(self):
        """
        weight initialize
        """
        for param in self.linears.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
                if self.diagonal_weight > 0.0:
                    param.data += self.diagonal_weight * \
                                  torch.diag(torch.ones(
                                      param.size(-1), dtype=torch.float))
                if self.symmetric_init:
                    param.data += param.data.T
                    # param.data /= 2.0
            else:
                constant_(param, 0)

    def _get_norm(self, eps):
        """
        batch/layer/instance normalization
        """
        if self.attention_type in ['linear', 'galerkin', 'global']:
            if self.norm_type == 'instance':
                self.norm_K = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
                self.norm_V = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
            elif self.norm_type == 'layer':
                self.norm_K = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
                self.norm_V = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
        else:
            if self.norm_type == 'instance':
                self.norm_K = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
                self.norm_Q = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
            elif self.norm_type == 'layer':
                self.norm_K = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
                self.norm_Q = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)

    @staticmethod
    def _get_layernorm(normalized_dim, n_head, **kwargs):
        """
        layer normalization
        """
        return nn.ModuleList(
            [copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs)) for _ in range(n_head)])

    @staticmethod
    def _get_instancenorm(normalized_dim, n_head, **kwargs):
        """
        instance normalization
        """
        return nn.ModuleList(
            [copy.deepcopy(nn.InstanceNorm1d(normalized_dim, **kwargs)) for _ in range(n_head)])


if __name__ == '__main__':
    Q = torch.ones([10, 100, 512])
    K = torch.ones([10, 100, 512])
    V = torch.ones([10, 100, 512])
    layer = SimpleAttention(n_head=8, d_model=512, norm_type='instance', norm_add=True)
    y = layer(Q, K, V)
    print(y)
