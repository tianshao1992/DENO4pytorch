#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/12/11 12:55
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : PointNets.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from point_layers import STNLayer, SimpleLayer

from configs import activation_dict


class PointNetfeature(nn.Module):
    def __init__(self, input_dim, scaling=1.0, activation='relu',
                 global_feat=True, input_transform=False, feature_transform=False):
        super(PointNetfeature, self).__init__()

        self.feature_size = int(64 * scaling)
        self.input_transform = input_transform
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.activation = activation_dict[activation]

        if self.input_transform:
            self.stn = STNLayer(input_dim=input_dim, scaling=scaling, activation=activation)
        else:
            self.stn = SimpleLayer(planes=[input_dim, self.feature_size, self.feature_size])
        self.conv1 = torch.nn.Conv1d(input_dim, self.feature_size, 1)
        self.conv2 = torch.nn.Conv1d(self.feature_size, self.feature_size * 2, 1)
        self.conv3 = torch.nn.Conv1d(self.feature_size * 2, self.feature_size * 16, 1)
        self.bn1 = nn.BatchNorm1d(self.feature_size)
        self.bn2 = nn.BatchNorm1d(self.feature_size * 2)
        self.bn3 = nn.BatchNorm1d(self.feature_size * 16)

        if self.feature_transform:
            self.fstn = STNLayer(input_dim=self.feature_size, scaling=scaling, activation=activation)

    def forward(self, x):
        n_pts = x.size()[2]
        if self.input_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            trans = self.stn(x)

        x = self.activation(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.feature_size * 16)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.feature_size * 16, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], dim=1), trans, trans_feat


class PointNetRegressor(nn.Module):
    def __init__(self, output_dim=2, scaling=1.0, activation='relu'):
        super(PointNetRegressor, self).__init__()
        self.output_dim = output_dim
        self.feature_size = int(64 * scaling)
        # self.feature_transform = feature_transform
        # self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(self.feature_size * (16 + 1), self.feature_size * 8, 1)
        self.conv2 = torch.nn.Conv1d(self.feature_size * 8, self.feature_size * 4, 1)
        self.conv3 = torch.nn.Conv1d(self.feature_size * 4, self.feature_size * 2, 1)
        self.conv4 = torch.nn.Conv1d(self.feature_size * 2, self.output_dim, 1)
        self.bn1 = nn.BatchNorm1d(self.feature_size * 8)
        self.bn2 = nn.BatchNorm1d(self.feature_size * 4)
        self.bn3 = nn.BatchNorm1d(self.feature_size * 2)
        self.activation = activation_dict[activation]

    def forward(self, x):
        batchsize = x.shape[0]
        n_pts = x.shape[2]
        # x, trans, trans_feat = self.feat(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = x.view(batchsize, n_pts, self.output_dim)
        return x


class BasicPointNet(nn.Module):
    ##### Point-cloud deep learning for prediction of fluid flow fields on irregular geometries (supervised learning) #####

    # Authors: Ali Kashefi (kashefi@stanford.edu) and Davis Rempe (drempe@stanford.edu)
    # Description: Implementation of PointNet for *supervised learning* of computational mechanics on domains with irregular geometries
    # Version: 1.0
    # Guidance: We recommend opening and running the code on **[Google Colab](https://research.google.com/colaboratory)** as a first try.

    def __init__(self, input_dim, output_dim,
                 scaling=1.0, activation='relu',
                 input_transform=True, feature_transform=True):
        super(BasicPointNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scaling = scaling
        self.activation = activation

        self.feature_layer = PointNetfeature(input_dim=input_dim, scaling=scaling, activation=activation,
                                             global_feat=False,
                                             input_transform=input_transform,
                                             feature_transform=feature_transform)

        self.regressor_layer = PointNetRegressor(output_dim=output_dim,
                                                 scaling=scaling, activation=activation)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, trans, trans_feature = self.feature_layer(x)
        output = self.regressor_layer(x)

        return output, trans, trans_feature


if __name__ == '__main__':
    NetModel = BasicPointNet(input_dim=3, output_dim=2, scaling=1.0, activation='relu',
                             input_transform=False, feature_transform=True)
    x = torch.randn(32, 1024, 3)
    output, trans, trans_feature = NetModel(x)
    print(output.shape)
