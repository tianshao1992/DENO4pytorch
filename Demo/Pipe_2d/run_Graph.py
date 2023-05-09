#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/5/8 23:54
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : run_Graph.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
from fno.FNOs import FNO2d
from cnn.ConvNets import UNet2d
from gnn.PointNets import BasicPointNet
from Utilizes.process_data import DataNormer
from Utilizes.visual_data import MatplotlibVision, TextLogger

import matplotlib.tri as tri
import matplotlib.pyplot as plt
import time
import os


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

        pred = netmodel(xx)[0]
        loss = lossfunc(pred, yy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1) / batch_size


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

            pred = netmodel(xx)[0]
            loss = lossfunc(pred, yy)
            valid_loss += loss.item()

    return valid_loss / (batch + 1) / batch_size


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
        pred = netmodel(xx)[0]

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), xx.cpu().numpy(), yy.numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################
    # 模型名称
    name = 'BasicPointNet'
    work_path = os.path.join('work', name)
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)
    # log文件记录
    Logger = TextLogger(filename=os.path.join(work_path, 'train.log'))

    # 计算设备
    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')
    Logger.info("Model Name: {:s}, Computing Device: {:s}".format(name, str(Device)))

    # 数据集路径
    INPUT_X = './data/Pipe_X.npy'
    INPUT_Y = './data/Pipe_Y.npy'
    OUTPUT_Sigma = './data/Pipe_Q.npy'

    # 训练集参数
    ntrain = 1000
    nvalid = 200
    r1 = 1
    r2 = 2
    s1 = int(((129 - 1) / r1) + 1)
    s2 = int(((129 - 1) / r2) + 1)

    ################################################################
    # load data
    ################################################################

    inputX = np.load(INPUT_X)
    inputX = torch.tensor(inputX, dtype=torch.float) / 10.
    inputY = np.load(INPUT_Y)
    inputY = torch.tensor(inputY, dtype=torch.float)
    input = torch.stack([inputX, inputY], dim=-1)

    output = np.load(OUTPUT_Sigma)[:, (0,)].squeeze()
    output = torch.tensor(output, dtype=torch.float).unsqueeze(-1)

    train_x = input[:ntrain, ::r1, ::r2][:, :s1, :s2].reshape((ntrain, -1, 2))
    train_y = output[:ntrain, ::r1, ::r2][:, :s1, :s2].reshape((ntrain, -1, 1))
    valid_x = input[ntrain:ntrain + nvalid, ::r1, ::r2][:, :s1, :s2].reshape((nvalid, -1, 2))
    valid_y = output[ntrain:ntrain + nvalid, ::r1, ::r2][:, :s1, :s2].reshape((nvalid, -1, 1))

    # x_normalizer = DataNormer(train_x.numpy(), method='mean-std', axis=(0,))
    # train_x = x_normalizer.norm(train_x)
    # valid_x = x_normalizer.norm(valid_x)
    #
    # y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    # train_y = y_normalizer.norm(train_y)
    # valid_y = y_normalizer.norm(valid_y)

    # 模型参数
    in_dim = 2
    out_dim = 1
    sampling_size = s1 * s2
    modes = (12, 12)
    width = 32
    depth = 4
    steps = 1
    padding = 8
    dropout = 0.0

    # 训练参数
    batch_size = 8
    epochs = 1000
    learning_rate = 0.0002
    scheduler_step = 800
    scheduler_gamma = 0.1

    Logger.info('Total epochs: {:d}, learning_rate: {:e}, scheduler_step: {:d}, scheduler_gamma: {:e}'
                .format(epochs, learning_rate, scheduler_step, scheduler_gamma))

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                               batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                               batch_size=batch_size, shuffle=False, drop_last=False)

    ################################################################
    # Neural Networks
    ################################################################

    # 建立网络
    if 'BasicPointNet' in name:
        Net_model = BasicPointNet(input_dim=in_dim, output_dim=out_dim, scaling=1.0, activation='gelu', max_pool=False,
                                  input_transform=True, feature_transform=True).to(Device)

    input = torch.randn(batch_size, train_x.shape[1], train_x.shape[2]).to(Device)
    model_statistics = summary(Net_model, input_data=input, device=Device, )
    Logger.write(str(model_statistics))

    # 损失函数
    Loss_func = nn.MSELoss()
    # Loss_func = FieldsLpLoss(size_average=False)
    # L1loss = nn.SmoothL1Loss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.8, 0.9), weight_decay=1e-6)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('s'))

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
        Logger.info('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
                    format(epoch, Optimizer.param_groups[0]['lr'], log_loss[0][-1], log_loss[1][-1],
                           time.time() - star_time))

        star_time = time.time()

        if epoch > 0 and epoch % 10 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(work_path, 'log_loss.svg'))
            plt.close(fig)

        ################################################################
        # Visualization
        ################################################################

        if epoch > 0 and epoch % 50 == 0:

            train_coord, train_prof, train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_coord, valid_prof, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))

            train_coord = train_coord.reshape((batch_size, s1, s2, -1))
            train_coord[..., 0] *= 10.
            train_true = train_true.reshape((batch_size, s1, s2, -1))
            train_pred = train_pred.reshape((batch_size, s1, s2, -1))
            valid_coord = valid_coord.reshape((batch_size, s1, s2, -1))
            valid_coord[..., 0] *= 10.
            valid_true = valid_true.reshape((batch_size, s1, s2, -1))
            valid_pred = valid_pred.reshape((batch_size, s1, s2, -1))

            for fig_id in range(batch_size):
                fig, axs = plt.subplots(1, 3, figsize=(18, 6), layout='constrained', num=2)
                Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], train_coord[fig_id])
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(batch_size):
                fig, axs = plt.subplots(1, 3, figsize=(18, 6), layout='constrained', num=3)
                Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], valid_coord[fig_id])
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)
