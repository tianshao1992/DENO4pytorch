#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/25 19:56
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : configs.py
"""
import sko
import torch

_optimizer_dict = \
                {'GA': sko.GA.GA,
                 'DE': sko.DE.DE,
                 'SA': sko.SA.SA,
                 'AFSA': sko.AFSA.AFSA,
                 'SGD': torch.optim.SGD,
                 'Adam': torch.optim.Adam,
                 'AdamW': torch.optim.AdamW,
                 None: sko.GA}

_heuristic_optimizer_dict = ['GA', 'DE', 'SA', 'ACA', 'AFSA', 'IA', None]
_gradient_optimizer_dict = ['SGD', 'Adam', 'AdamW']

def default(value, d):
    """
        helper taken from https://github.com/lucidrains/linear-attention-transformer
    """
    return d if value is None else value
