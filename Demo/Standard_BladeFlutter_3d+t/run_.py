#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/7/16 19:18
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : run_.py
# @Description    : 标准化的数据读入、处理、训练、测试、评估
"""

import os
import sys
import time
import numpy as np
import torch

from operator import add
from functools import reduce

from Datasets.file_reader.reader import DataReader
from Datasets.basic_data.image_field import ImageField
from Datasets.basic_data.scalar_data import ScalarData
from Datasets.data_loader.loader import CustomDataset

def data_preprocess(reader, train_size):
    """
        data preprocess
        :param file_loader: Mat loader
    """
    design = np.transpose(reader.read_field('Aphis_'), (1, 0))
    fields = np.transpose(reader.read_field('fields'), (5, 3, 0, 1, 4, 2))[:, :, :, ::2]  # (N, T, H, W, I, F)
    coords_x = fields[..., (0, 1, 2)]
    fields = fields[..., (3, 4)]
    target = np.transpose(reader.read_field('MFs'), (2, 0, 1))

    coords_t = torch.linspace(0, 2*np.pi, fields.shape[1], dtype=torch.float32)[None, :, None, None, None]\
                .repeat((fields.shape[0],) + (1,) + tuple(fields.shape[2:-1])).unsqueeze(-1)
    coords = torch.cat((coords_x, coords_t), dim=-1)

    coords = coords.reshape(tuple(coords.shape[0:-2]) + (-1,))
    fields = fields.reshape(tuple(fields.shape[0:-2]) + (-1,))

    index = train_test_split(design, train_size)

    return design, fields, coords, target, index

def train_test_split(design, train_size):
    """
        train test split
    """
    train_design = {4: [0, 90, 180, 270],
                   7: [0, 30, 90, 150, 180, 270, 340],
                   10: [0, 30, 80, 90, 150, 180, 240, 270, 300, 340],}
    test_design = [60, 120, 130, 210, 330, 350]
    train_index = []
    test_index = []
    for i in range(len(design)):
        if int(design[i]) in train_design[train_size]:
            train_index.append(i)
        elif int(design[i]) in test_design:
            test_index.append(i)
    return train_index, test_index



if __name__ == '__main__':
    ################################################################
    # load data
    ################################################################

    data_path = os.path.join('data', 'y_bend-10_RN1_A5_APhis_trans_field.mat')
    reader = DataReader(data_path)

    design, fields, coords, target, index = data_preprocess(reader, train_size=10)

    design = ScalarData(data=design, name=['design'])
    fields_name = reduce(add, [['P_' + str(i), 'M_' + str(i)] for i in range(-2, 3)])
    fields = ImageField(data=fields, grid=coords, name=fields_name)
    target_name = [['MFs_' + str(i)] for i in range(-2, 3)]
    target = ScalarData(data=target, name=target_name)


    train_index, test_index = index[0], index[1]
    TrainDataset = CustomDataset(design[train_index], fields[train_index], target=target[train_index])





