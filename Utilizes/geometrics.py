#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/5/9 16:05
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : geometrics.py
"""

import numpy as np
import torch
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange


def ccw_sort(points):
    """Sort given polygon points in CCW order"""
    points = np.array(points)
    mean = np.mean(points, axis=0)
    coords = points - mean
    s = np.arctan2(coords[:, 0], coords[:, 1])
    return points[np.argsort(s), :]


def index_points(points, idx):
    """
    find idx of each batch points
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


# helper
def knn(x1, x2, k):
    """
    caculate topk distance from x1 to x2
        Input:
        x1: input points data, [b, n1, c]
        x2: input points data, [b, n2, c]
        k: int
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    # x1 [b, n1, c], x2 [b, n2, c]
    inner = -2 * torch.matmul(x1, rearrange(x2, 'b n c -> b c n'))  # [b n1 n2]
    xx = torch.sum(x1 ** 2, dim=-1, keepdim=True)  # [b, n1, 1]
    yy = torch.sum(x2 ** 2, dim=-1, keepdim=True)  # [b, n2, 1]
    pairwise_distance = -xx - inner - yy.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, n1, k)
    return idx
