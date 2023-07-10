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


def gen_uniform_grid(x):
    """
    Args:
        x: input coordinates
    Returns:
        res: input transform
    """
    shape = x.shape
    if len(shape) == 3:

        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float32)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(x.device)

    elif len(shape) == 4:

        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float32)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float32)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.linspace(0, 1, size_z, dtype=torch.float32)
        gridz = gridz.reshape(1, 1, 1, size_y, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(x.device)

    elif len(shape) == 5:
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float32)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float32)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.linspace(0, 1, size_z, dtype=torch.float32)
        gridz = gridz.reshape(1, 1, 1, size_y, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(x.device)
    else:
        raise NotImplementedError("The tensor shape is Not implemented yet!")