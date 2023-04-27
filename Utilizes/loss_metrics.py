#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/6 17:37
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : utilize.py
"""
import os

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

activation_dict = \
    {'gelu': nn.GELU(), 'silu': nn.SiLU(), 'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'leakyrelu': nn.LeakyReLU(),
     None: nn.ReLU()}

additional_attr = ['normalizer', 'raw_laplacian', 'return_latent',
                   'residual_type', 'norm_type', 'norm_eps', 'boundary_condition',
                   'upscaler_size', 'downscaler_size', 'spacial_dim', 'spacial_fc',
                   'regressor_activation', 'attn_activation',
                   'downscaler_activation', 'upscaler_activation',
                   'encoder_dropout', 'decoder_dropout', 'ffn_dropout']


def default(value, d):
    """
        helper taken from https://github.com/lucidrains/linear-attention-transformer
    """
    return d if value is None else value


class CarDataset(Dataset):
    """
    :param results, list
    :param profile_skip, int
    """

    def __init__(self, results, seq_len=40000, profile_skip=2):
        self.results = results
        self.seq_len = seq_len
        self.profile_skip = profile_skip

    def __getitem__(self, idx):
        # 根据 idx 取出其中一个
        _, Re, q, x, f, c, r = self.results[idx]
        n_points = min(x.shape[0], self.seq_len)
        Re = np.tile(np.array(Re).reshape((1, -1)), [self.seq_len, 1]).astype(np.float32) / 2.0e7  # Re normalization
        Q = np.tile(q[::self.profile_skip, :].reshape((1, -1)), [self.seq_len, 1]).astype(np.float32)
        Q = np.concatenate((Re, Q), axis=1)
        f = f[:, (0, 1, -1)]
        index = np.random.choice(n_points, self.seq_len)
        X = x[index].astype(np.float32)
        F = f[index].astype(np.float32)
        return Q, X, F

    def get_single(self, idx):
        Q, X, F = [], [], []
        for id in idx:
            _, re, q, x, f, c, r = self.results[id]
            n_points = x.shape[0]
            re = np.tile(np.array(re).reshape((1, -1)), [n_points, 1]).astype(np.float32) / 2.0e7  # Re normalization
            q = np.tile(q[::self.profile_skip, :].reshape((1, -1)), [n_points, 1]).astype(np.float32)
            Q.append(torch.tensor(np.concatenate((re, q), axis=1), dtype=torch.float32))
            X.append(torch.tensor(x, dtype=torch.float32))
            F.append(torch.tensor(f[:, (0, 1, -1)], dtype=torch.float32))
        return Q, X, F

    def __len__(self):  # 总数据的多少
        return len(self.results)


# loss function with rel/abs Lp loss
class FieldsLpLoss(object):
    def __init__(self, d=2, p=2, reduction=True, size_average=False):
        super(FieldsLpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):

        # Assume uniform mesh
        # h = 1.0 / (x.size()[1] - 1.0) (h ** (self.d / self.p)) *

        if torch.is_tensor(x):

            all_norms = torch.norm(x.reshape(x.shape[0], -1, x.shape[-1]) - y.reshape(x.shape[0], -1, x.shape[-1]), self.p, 1)

            if self.reduction:
                if self.size_average:
                    return torch.mean(all_norms)
                else:
                    return all_norms
        else:
            all_norms = np.linalg.norm(x.reshape(x.shape[0], -1, x.shape[-1]) - y.reshape(x.shape[0], -1, x.shape[-1]),
                                   self.p, 1)

            if self.reduction:
                if self.size_average:
                    return np.mean(all_norms)
                else:
                    return all_norms

        return all_norms

    def rel(self, x, y):

        if torch.is_tensor(x):
            diff_norms = torch.norm(x.reshape(x.shape[0], -1, x.shape[-1]) - y.reshape(x.shape[0], -1, x.shape[-1]), self.p, 1)
            y_norms = torch.norm(y.reshape(x.shape[0], -1, x.shape[-1]), self.p, 1)

            if self.reduction:
                if self.size_average:
                    return torch.mean(diff_norms / y_norms)
                else:
                    return diff_norms / y_norms
        else:
            diff_norms = np.linalg.norm(x.reshape(x.shape[0], -1, x.shape[-1]) - y.reshape(x.shape[0], -1, x.shape[-1]),
                                    self.p, 1)
            y_norms = np.linalg.norm(y.reshape(x.shape[0], -1, x.shape[-1]), self.p, 1)

            if self.reduction:
                if self.size_average:
                    return np.mean(diff_norms / (y_norms+1e-20))
                else:
                    return diff_norms / (y_norms+1e-20)

        return diff_norms / (y_norms+1e-20)

    def __call__(self, x, y):
        return self.rel(x, y)
