#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/7/9 23:07
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : optim_FNO.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Utilizes.process_data import MatLoader
from fno.FNOs import FNO2d
from basic.basic_layers import FcnSingle
from cnn.ConvNets import DownSampleNet2d
from Utilizes.process_data import DataNormer
from Utilizes.visual_data import MatplotlibVision, TextLogger

import matplotlib.pyplot as plt
import time
import os
from torchinfo import summary
from shutil import copyfile
from operator import add
from functools import reduce

from utilize import predictor, train, valid, inference, data_preprocess, custom_dataset


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'MLP'
    work_path = os.path.join('work', name)
    train_path = os.path.join(work_path, 'train')
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)
        os.makedirs(train_path)

    # 将控制台的结果输出到log文件
    Logger = TextLogger(os.path.join(train_path, 'train.log'))

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')
    Logger.info("Model Name: {:s}, Computing Device: {:s}".format(name, str(Device)))

    ntrain = 20
    nvalid = 5

    epochs = 201
    learning_rate = 0.001
    scheduler_step = 151
    scheduler_gamma = 0.1

    Logger.info('Total epochs: {:d}, learning_rate: {:e}, scheduler_step: {:d}, scheduler_gamma: {:.3e}'
                .format(epochs, learning_rate, scheduler_step, scheduler_gamma))

    ################################################################
    # load data
    ################################################################
    data_path = 'data/y_bend-10_RN1_A5_APhis_trans_field.mat'
    reader = MatLoader(data_path)
    design, _, _, target = data_preprocess(reader, seed=1000)

    design_normer = DataNormer(design)
    target_normer = DataNormer(target, method='mean-std')
    design = design_normer.norm(design)
    target = target_normer.norm(target)

    from utilize import cal_damps
    plt.figure(10)
    plt.clf()
    plt.scatter(design_normer.back(design[:ntrain]),
                cal_damps(target_normer.back(target[:ntrain])).min(axis=-1), label='train')
    plt.scatter(design_normer.back(design[-nvalid:]),
                cal_damps(target_normer.back(target[-nvalid:])).min(axis=-1), label='valid')
    plt.legend()
    plt.savefig(os.path.join(work_path, 'train_test_split.jpg'))

    del reader

    Net_model = FcnSingle(planes=(1, 32, 32, 1), last_activation=True).to(Device)

    # 损失函数
    Loss_func = nn.MSELoss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=0.0)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    field_name = reduce(add, [['P_' + str(i), 'M_' + str(i)] for i in range(1, 6)])
    Visual = MatplotlibVision(train_path, input_name=('x', 'y'), field_name=field_name)

    ################################################################

    for epoch in range(epochs):

        target_pred = Net_model(design.to(Device))
        target_loss = Loss(target_pred, target.to(Device))
        Optimizer.zero_grad()
        target_loss.backward()
        Optimizer.step()

        train_loss[0] += fields_loss.item() * input_sizes[0]
        train_loss[1] += target_loss.item() * input_sizes[0]
        total_size += input_sizes[0]
        history_loss.append([fields_loss.item(), target_loss.item()])

        scheduler.step()



