#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/12/11 12:55
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : PointNets.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import activation_dict


class STNLayer(nn.Module):
    '''
    1维谱卷积
    Modified Zongyi Li's Spectral1dConv code
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py
    '''

    def __init__(self, input_dim, scaling=1.0, activation='relu'):
        super(STNLayer, self).__init__()

        self.input_dim = input_dim
        self.feature_size = int(64 * scaling)
        self.conv1 = nn.Conv1d(input_dim, self.feature_size, 1)
        self.conv2 = nn.Conv1d(self.feature_size, self.feature_size * 2, 1)
        self.conv3 = nn.Conv1d(self.feature_size * 2, self.feature_size * 16, 1)
        self.fc1 = nn.Linear(self.feature_size * 16, self.feature_size * 8)
        self.fc2 = nn.Linear(self.feature_size * 8, self.feature_size * 4)
        self.fc3 = nn.Linear(self.feature_size * 4, input_dim ** 2)
        self.activation = activation_dict[activation]

        self.bn1 = nn.BatchNorm1d(self.feature_size)
        self.bn2 = nn.BatchNorm1d(self.feature_size * 2)
        self.bn3 = nn.BatchNorm1d(self.feature_size * 16)
        self.bn4 = nn.BatchNorm1d(self.feature_size * 8)
        self.bn5 = nn.BatchNorm1d(self.feature_size * 4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.feature_size * 16)

        x = self.activation(self.bn4(self.fc1(x)))
        x = self.activation(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.input_dim, dtype=torch.float32).view(1, self.input_dim ** 2).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.input_dim, self.input_dim)
        return x


class SimpleLayer(nn.Module):

    def __init__(self, planes: list, activation='relu', last_activation=False):
        super(SimpleLayer, self).__init__()

        self.planes = planes
        self.active = activation_dict[activation]

        self.layers = nn.ModuleList()
        for i in range(len(self.planes) - 2):
            self.layers.append(nn.Conv1d(self.planes[i], self.planes[i + 1], 1))
            self.layers.append(nn.BatchNorm1d(self.planes[i + 1]))
            self.layers.append(self.active)
        self.layers.append(nn.Conv1d(self.planes[-2], self.planes[-1], 1))
        self.layers.append(nn.BatchNorm1d(self.planes[-1]))
        if last_activation:
            self.layers.append(self.active)
        self.layers = nn.Sequential(*self.layers)  # *的作用是解包

    def forward(self, in_var):
        """
        forward compute
        :param in_var: (batch_size, ..., input_dim)
        """
        out_var = self.layers(in_var)
        return out_var


if __name__ == '__main__':
    stn = STNLayer(3)
    x = torch.randn(32, 3, 1024)
    print(stn(x).shape)

    stn = STNLayer(64)
    x = torch.randn(32, 64, 1024)
    print(stn(x).shape)

    stn = SimpleLayer([3, 64, 64], last_activation=True)
    x = torch.randn(32, 3, 1024)
    print(stn(x).shape)
