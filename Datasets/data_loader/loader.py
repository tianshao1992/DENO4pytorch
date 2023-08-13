#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/7/16 20:00
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : loader.py
# @Description    : 以basic_data实现的数据加载器
"""

from Datasets.basic_data._base_data import BasicData
import torch
from torch.utils.data import Dataset
import torch_geometric.datasets

class CustomDataset(Dataset):
    """
    以basic_data实现的数据加载器
    """
    def __init__(self, input: BasicData, output: BasicData, **kwargs):

        self.input = input
        self.output = output
        self.data_name = ['input', 'output']
        self.sample_size = input.sample_size

        if self.output.sample_size != self.sample_size:
            raise ValueError("sample_size must be the same for input and output data")

        for key, value in kwargs.items():
            setattr(self, key, value)
            self.data_name.append(key)
            if self.sample_size != value.sample_size:
                raise ValueError("sample_size must be the same for all input data")

    def __getitem__(self, idx):  # 根据 idx 取出其中一个

        single_data = []
        for key in self.data_name:
            single_data.append(self.__dict__[key][idx])
        return single_data

    def __len__(self):  # 总数据的多少
        return self.sample_size
