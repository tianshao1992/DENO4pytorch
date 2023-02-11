#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/2/11 2:35
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : run_Trans.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
from Utilizes.process_data import DataNormer, MatLoader
from transformer.Transformers import SimpleTransformer
from Utilizes.visual_data import MatplotlibVision, TextLogger

import matplotlib.pyplot as plt
import time
import os
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
    return torch.cat((gridx, gridy), dim=-1).to(x.device)


def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    """
    Args:
        data_loader: output fields at last time step
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """

    train_loss = 0
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)
        grid, edge = feature_transform(xx)

        pred = netmodel(xx, grid, edge, grid)['preds']
        loss = lossfunc(pred, yy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1)


def valid(dataloader, netmodel, device, lossfunc):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """
    valid_loss = 0
    with torch.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            xx = xx.to(device)
            yy = yy.to(device)
            grid, edge = feature_transform(xx)

            pred = netmodel(xx, grid, edge, grid)['preds']
            loss = lossfunc(pred, yy)
            valid_loss += loss.item()

    return valid_loss / (batch + 1)


def inference(dataloader, netmodel, device):
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """
    with torch.no_grad():
        xx, yy = next(iter(dataloader))
        xx = xx.to(device)
        grid, edge = feature_transform(xx)
        pred = netmodel(xx, grid, edge, grid)['preds']

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), grid.cpu().numpy(), yy.numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'FNO'
    work_path = os.path.join('work', name)
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    # 将控制台的结果输出到log文件
    sys.stdout = TextLogger(os.path.join(work_path, 'train.log'), sys.stdout)

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    train_file = os.path.join('data', 'standard', 'piececonst_r421_N1024_smooth1.mat')
    valid_file = os.path.join('data', 'standard', 'piececonst_r421_N1024_smooth1.mat')

    in_dim = 1
    out_dim = 1
    ntrain = 1000
    nvalid = 100

    modes = (12, 12)
    width = 32
    depth = 4
    steps = 1
    padding = 9
    dropout = 0.0

    batch_size = 32
    batch_size2 = batch_size

    epochs = 500
    learning_rate = 0.001
    scheduler_step = 400
    scheduler_gamma = 0.1

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    r_train = 5
    h_train = int(((421 - 1) / r_train) + 1)
    s_train = h_train

    r_valid = 5
    h_valid = int(((421 - 1) / r_valid) + 1)
    s_valid = h_valid

    ################################################################
    # load data
    ################################################################

    reader = MatLoader(train_file)
    train_x = reader.read_field('coeff')[:ntrain, ::r_train, ::r_train][:, :s_train, :s_train, None]
    train_y = reader.read_field('sol')[:ntrain, ::r_train, ::r_train][:, :s_train, :s_train, None]

    valid_x = reader.read_field('coeff')[-nvalid:, ::r_valid, ::r_valid][:, :s_valid, :s_valid, None]
    valid_y = reader.read_field('sol')[-nvalid:, ::r_valid, ::r_valid][:, :s_valid, :s_valid, None]

    del reader

    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)

    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                               batch_size=batch_size, shuffle=False, drop_last=True)

    ################################################################
    #  Neural Networks
    ################################################################
    with open(os.path.join('transformer_config.yml')) as f:
        config = yaml.full_load(f)
    config = config['Burgers_1d']

    # 建立网络
    Net_model = SimpleTransformer(**config).to(Device)

    # input1 = torch.randn(batch_size, train_x.shape[1], train_x.shape[2], train_x.shape[3]).to(Device)
    # input2 = torch.randn(batch_size, train_x.shape[1], train_x.shape[2], 2).to(Device)
    # print(name)
    # summary(Net_model, input_data=[input1, input2], device=Device)

    # 损失函数
    Loss_func = nn.MSELoss()
    # L1loss = nn.SmoothL1Loss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-4)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('f',))

    star_time = time.time()
    log_loss = [[], []]

    ################################################################
    # train process
    ################################################################

    for epoch in range(epochs):

        Net_model.train()
        log_loss[0].append(train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))

        Net_model.eval()
        log_loss[1].append(valid(valid_loader, Net_model, Device, Loss_func))
        print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, learning_rate, log_loss[0][-1], log_loss[1][-1], time.time() - star_time))

        star_time = time.time()

        if epoch > 0 and epoch % 20 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(work_path, 'log_loss.svg'))
            plt.close(fig)

        ################################################################
        # Visualization
        ################################################################

        if epoch > 0 and epoch % 20 == 0:
            # print('epoch: {:6d}, lr: {:.3e}, eqs_loss: {:.3e}, bcs_loss: {:.3e}, cost: {:.2f}'.
            #       format(epoch, learning_rate, log_loss[-1][0], log_loss[-1][1], time.time()-star_time))
            train_source, train_coord, train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_source, valid_coord, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))

            for fig_id in range(10):
                fig, axs = plt.subplots(1, 3, figsize=(18, 5), layout='constrained', num=2)
                Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], train_coord[fig_id])
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(10):
                fig, axs = plt.subplots(1, 3, figsize=(18, 5), layout='constrained', num=3)
                Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], valid_coord[fig_id])
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)
