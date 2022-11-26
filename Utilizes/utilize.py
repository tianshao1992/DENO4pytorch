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
        Re = np.tile(np.array(Re).reshape((1, -1)), [self.seq_len, 1]).astype(np.float32) / 2.0e7 # Re normalization
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

class DataNormer():
    """
        data normalization at last dimension
    """
    def __init__(self, data, method="min-max"):
        axis = tuple(range(len(data.shape) - 1))
        self.method = method
        if method == "min-max":
            self.max = np.max(data, axis=axis)
            self.min = np.min(data, axis=axis)

        elif method == "mean-std":
            self.mean = np.mean(data, axis=axis)
            self.std = np.std(data, axis=axis)

    def norm(self, x):
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = 2 * (x - torch.tensor(self.min, device=x.device)) \
                    / (torch.tensor(self.max, device=x.device) - torch.tensor(self.min, device=x.device)) - 1
            elif self.method == "mean-std":
                x = (x - torch.tensor(self.mean, device=x.device)) / (torch.tensor(self.std, device=x.device))
        else:
            if self.method == "min-max":
                x = 2 * (x - self.min) / (self.max - self.min) - 1
            elif self.method == "mean-std":
                x = (x - self.mean) / (self.std)

        return x

    def back(self, x):
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = (x + 1) / 2 * (torch.tensor(self.max, device=x.device)
                                   - torch.tensor(self.min, device=x.device)) + torch.tensor(self.min, device=x.device)
            elif self.method == "mean-std":
                x = x * (torch.tensor(self.std, device=x.device)) + torch.tensor(self.mean, device=x.device)
        else:
            if self.method == "min-max":
                x = (x + 1) / 2 * (self.max - self.min) + self.min
            elif self.method == "mean-std":
                x = x * (self.std) + self.mean
        return x