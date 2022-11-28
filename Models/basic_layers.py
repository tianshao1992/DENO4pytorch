#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/6 17:37
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : basic_layers.py
"""

import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from Models.configs import *


class FcnSingle(nn.Module):
    def __init__(self, planes: list, activation="gelu"):
        # =============================================================================
        #     Inspired by M. Raissi a, P. Perdikaris b,∗, G.E. Karniadakis.
        #     "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems
        #     involving nonlinear partial differential equations".
        #     Journal of Computational Physics.
        # =============================================================================
        super(FcnSingle, self).__init__()
        self.planes = planes
        self.active = activation_dict[activation]

        self.layers = nn.ModuleList()
        for i in range(len(self.planes) - 2):
            self.layers.append(nn.Linear(self.planes[i], self.planes[i + 1]))
            self.layers.append(self.active)
        self.layers.append(nn.Linear(self.planes[-2], self.planes[-1]))
        self.layers = nn.Sequential(*self.layers)
        self.reset_parameters()

    def reset_parameters(self):
        """
        weight initialize
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.xavier_uniform_(m.weight, gain=1)
                m.bias.data.zero_()

    def forward(self, in_var):
        """
        forward compute
        :param in_var: (batch_size, ..., input_dim)
        """
        out_var = self.layers(in_var)
        return out_var


class FcnMulti(nn.Module):
    def __init__(self, planes: list, activation="gelu"):
        # =============================================================================
        #     Inspired by Haghighat Ehsan, et all.
        #     "A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics"
        #     Computer Methods in Applied Mechanics and Engineering.
        # =============================================================================
        super(FcnMulti, self).__init__()
        self.planes = planes
        self.active = activation_dict[activation]

        self.layers = nn.ModuleList()
        for j in range(self.planes[-1]):
            layer = []
            for i in range(len(self.planes) - 2):
                layer.append(nn.Linear(self.planes[i], self.planes[i + 1]))
                layer.append(self.active)
            layer.append(nn.Linear(self.planes[-2], 1))
            self.layers.append(nn.Sequential(*layer))
        self.reset_parameters()

    def reset_parameters(self):
        """
        weight initialize
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.xavier_uniform_(m.weight, gain=1)
                m.bias.data.zero_()

    def forward(self, in_var):
        """
        forward compute
        :param in_var: (batch_size, ..., input_dim)
        """
        y = []
        for i in range(self.planes[-1]):
            y.append(self.layers[i](in_var))
        return torch.cat(y, dim=-1)


class DeepONetMulti(nn.Module):
    # =============================================================================
    #     Inspired by L. Lu, J. Pengzhan, G.E. Karniadakis.
    #     "DeepONet: Learning nonlinear operators for identifying differential equations based on
    #     the universal approximation theorem of operators".
    #     arXiv:1910.03193v3 [cs.LG] 15 Apr 2020.
    # =============================================================================
    def __init__(self, input_dim: int, operator_dims: list, output_dim: int,
                 planes_branch: list, planes_trunk: list, activation='gelu'):
        """
        :param input_dim: int, the coordinates dim for trunk net
        :param operator_dims: list，the operate dims list for each branch net
        :param output_dim: int, the predicted variable dims
        :param planes_branch: list, the hidden layers dims for branch net
        :param planes_trunk: list, the hidden layers dims for trunk net
        :param operator_dims: list，the operate dims list for each branch net
        :param activation: activation function
        """
        super(DeepONetMulti, self).__init__()
        self.branches = nn.ModuleList()
        self.trunks = nn.ModuleList()
        for dim in operator_dims:
            self.branches.append(FcnSingle([dim] + planes_branch, activation=activation))
        for _ in range(output_dim):
            self.trunks.append(FcnSingle([input_dim] + planes_trunk, activation=activation))

        self.reset_parameters()

    def reset_parameters(self):
        """
        weight initialize
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.xavier_uniform_(m.weight, gain=1)
                m.bias.data.zero_()

    def forward(self, u_vars, y_var):
        """
        forward compute
        :param in_var: (batch_size, ..., input_dim)
        """
        B = 1.
        for u_var, branch in zip(u_vars, self.branches):
            B *= branch(u_var)
        out_var = []
        for trunk in self.trunks:
            T = trunk(y_var)
            out_var.append(torch.sum(B * T, dim=-1))
        out_var = torch.stack(out_var, dim=-1)
        return out_var


class Identity(nn.Module):
    '''
    a placeholder layer similar to tensorflow.no_op():
    https://github.com/pytorch/pytorch/issues/9160#issuecomment-483048684
    not used anymore as
    https://pytorch.org/docs/stable/generated/torch.nn.Identity.html
    edge and grid are dummy inputs
    '''

    def __init__(self, in_features=None, out_features=None,
                 *args, **kwargs):
        super(Identity, self).__init__()

        if in_features is not None and out_features is not None:
            self.id = nn.Linear(in_features, out_features)

        else:
            self.id = nn.Identity()

    def forward(self, x, edge=None, grid=None):
        """
        forward compute
        :param in_var: (batch_size, input_dim, ...)
        """
        # todo: 利用 einsteinsum 构造
        if len(x.shape) == 5:
            '''
            (-1, in, H, W, S) -> (-1, out, H, W, S)
            Used in SimpleResBlock
            '''
            x = x.permute(0, 2, 3, 4, 1)
            x = self.id(x)
            x = x.permute(0, 4, 1, 2, 3)
        elif len(x.shape) == 4:
            '''
            (-1, in, H, W) -> (-1, out, H, W)
            Used in SimpleResBlock
            '''
            x = x.permute(0, 2, 3, 1)
            x = self.id(x)
            x = x.permute(0, 3, 1, 2)

        elif len(x.shape) == 3:
            '''
            (-1, S, in) -> (-1, S, out)
            Used in SimpleResBlock
            '''
            # x = x.permute(0, 2, 1)
            x = self.id(x)
            # x = x.permute(0, 2, 1)
        elif len(x.shape) == 2:
            '''
            (-1, in) -> (-1, out)
            Used in SimpleResBlock
            '''
            x = self.id(x)
        else:
            raise NotImplementedError("input sizes not implemented.")

        return x




if __name__ == '__main__':
    x = torch.ones([10, 64, 64, 3])
    layer = FcnSingle([3, 64, 64, 10])
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 64, 64, 3])
    layer = FcnMulti([3, 64, 64, 10])
    y = layer(x)
    print(y.shape)

    us = [torch.ones([10, 256 * 2]), torch.ones([10, 1])]
    x = torch.ones([10, 2])
    layer = DeepONetMulti(input_dim=2, operator_dims=[256 * 2, 1], output_dim=5,
                          planes_branch=[64] * 3, planes_trunk=[64] * 2)
    y = layer(us, x)
    print(y.shape)


