#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/2/13 0:17
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : run_Trans.py
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Utilizes.process_data import MatLoader
from Utilizes.loss_metrics import FieldsLpLoss
from transformer.Transformers import FourierTransformer2D
from Utilizes.visual_data import MatplotlibVision, TextLogger

import matplotlib.pyplot as plt
import time
import os
from torchinfo import summary
import sys
import yaml


def feature_transform(x):
    """
    Args:
        x: input coordinates
    Returns:
        res: input transform
    """
    shape = x.shape
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.linspace(0, 1, size_x, dtype=torch.float32)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.linspace(0, 1, size_y, dtype=torch.float32)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])

    edge = torch.ones((x.shape[0], 1))
    return torch.cat((gridx, gridy), dim=-1).to(x.device), edge.to(x.device)

def valid(dataloader, netmodel, device, lossfunc):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """
    valid_loss = []
    with torch.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            xx = xx.to(device)
            yy = yy.to(device)

            grid, edge = feature_transform(xx)
            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = netmodel(xx, grid, edge, grid)['preds']
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
                xx = torch.cat((xx[..., step:], im), dim=-1)

            loss = lossfunc(pred, yy)
            valid_loss.append(loss)
        valid_loss = torch.concat(valid_loss, dim=0)

    return valid_loss


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'Trans'
    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    # train_file = './data/ns_V1e-3_N5000_T50.mat'
    train_file = './data/ns_V1e-5_N1200_T20.mat'

    in_dim = 10
    out_dim = 1
    # ntrain = 4000
    # nvalid = 1000

    for ntrain in (50, 100, 200, 400, 600, 800, 1000):
    # ntrain = 600
        nvalid = 200

        work_path = os.path.join('work', name, 'train_size-' + str(ntrain))
        isCreated = os.path.exists(work_path)
        if not isCreated:
            os.makedirs(work_path)

        # 将控制台的结果输出到log文件
        sys.stdout = TextLogger(os.path.join(work_path, 'test.log'), sys.stdout)

        modes = (20, 20)  # fno
        steps = 1  # fno
        padding = 8  # fno
        width = 32  # all
        depth = 4  # all
        dropout = 0.0

        batch_size = 4
        epochs = 500
        learning_rate = 0.001
        scheduler_step = 400
        scheduler_gamma = 0.1

        print(ntrain, epochs, learning_rate, scheduler_step, scheduler_gamma)

        sub = 1
        S = 64
        T_in = 10
        T = 10
        step = 1

        ################################################################
        # load data
        ################################################################

        reader = MatLoader(train_file)
        train_x = reader.read_field('u')[:ntrain, ::sub, ::sub, :T_in]
        train_y = reader.read_field('u')[:ntrain, ::sub, ::sub, T_in:T + T_in]

        valid_x = reader.read_field('u')[ntrain:, ::sub, ::sub, :T_in]
        valid_y = reader.read_field('u')[ntrain:, ::sub, ::sub, T_in:T + T_in]
        del reader

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                                   batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                                   batch_size=batch_size, shuffle=False, drop_last=True)

        ################################################################
        # Neural Networks
        ################################################################

        with open(os.path.join('transformer_config.yml')) as f:
            config = yaml.full_load(f)

        config = config['Turbulence_2d+t']

        # 建立网络
        Net_model = FourierTransformer2D(**config).to(Device)
        # input1 = torch.randn(batch_size, train_x.shape[1], train_x.shape[2], train_x.shape[3]).to(Device)
        # input2 = torch.randn(batch_size, train_x.shape[1], train_x.shape[2], 2).to(Device)
        # print(name)

        # (xx, yy) = next(iter(train_loader))
        # xx = xx.to(Device)
        # yy = yy.to(Device)
        # grid, edge = feature_transform(xx)
        # summary(Net_model, input_data=[xx, grid, edge, grid], device=Device)

        # 模型载入
        checkpoint = torch.load(os.path.join(work_path, 'latest_model.pth'))
        Net_model.load_state_dict(checkpoint['net_model'])
        # 损失函数
        Loss_metric = FieldsLpLoss()
        # 可视化
        Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('f',))

        star_time = time.time()
        log_loss = [[], []]

        ################################################################
        # statistics process
        ################################################################

        Net_model.eval()
        metric_l2 = valid(valid_loader, Net_model, Device, Loss_metric)

        print(metric_l2.mean())
