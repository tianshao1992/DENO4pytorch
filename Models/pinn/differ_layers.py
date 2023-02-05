#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/25 23:32
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : Differ_layers.py
"""

import torch
from torch.autograd import grad


def gradients(y, x):
    """
    计算y对x的一阶导数，dydx
    :param y: torch tensor，网络输出，shape（...×N）
    :param x: torch tensor，网络输入，shape（...×M）
    :return dydx: torch tensor，网络输入，shape（...M×N）如果N=1，则最后一个维度被缩并
    """
    return torch.stack([grad([y[..., i].sum()], [x], retain_graph=True, create_graph=True)[0]
                        for i in range(y.size(-1))], dim=-1).squeeze(-1)
