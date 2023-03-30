#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/2/7 15:20
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : DeepONets.py
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
from basic.basic_layers import *
from basic_layers import *
from Models.configs import *


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

    def forward(self, u_vars, y_var, size_set=True):
        """
        forward compute
        :param u_vars: tensor list[(batch_size, ..., operator_dims[0]), (batch_size, ..., operator_dims[1]), ...]
        :param y_var: (batch_size, ..., input_dim)
        :param size_set: bool, true for standard inputs, false for reduce points number in operator inputs
        """
        B = 1.
        for u_var, branch in zip(u_vars, self.branches):
            B *= branch(u_var)
        if not size_set:
            B_size = list(y_var.shape[1:-1])
            for i in range(len(B_size)):
                B = B.unsqueeze(1)
            B = torch.tile(B, [1, ] + B_size + [1, ])
        out_var = []
        for trunk in self.trunks:
            T = trunk(y_var)
            out_var.append(torch.sum(B * T, dim=-1))
        out_var = torch.stack(out_var, dim=-1)
        return out_var


if __name__ == "__main__":
    us = [torch.ones([10, 256 * 2]), torch.ones([10, 1])]
    x = torch.ones([10, 2])
    layer = DeepONetMulti(input_dim=2, operator_dims=[256 * 2, 1], output_dim=5,
                          planes_branch=[64] * 3, planes_trunk=[64] * 2)
    y = layer(us, x)
    print(y.shape)
