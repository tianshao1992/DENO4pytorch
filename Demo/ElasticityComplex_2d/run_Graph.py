#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/5/8 12:56
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
from Utilizes.geometrics import ccw_sort

import matplotlib.tri as tri
import matplotlib.pyplot as plt
import time
import os

from utils import readFire

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
    for batch, (xx, yy, bb) in enumerate(dataloader):
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
        for batch, (xx, yy, bb) in enumerate(dataloader):
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
        xx, yy, bb = next(iter(dataloader))
        xx = xx.to(device)
        pred = netmodel(xx)[0]

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), bb.cpu().numpy(), yy.numpy(), pred.cpu().numpy()


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
    datafile = './data/Data.npy'

    # 训练集参数
    ntrain = 300
    nvalid = 50

    ################################################################
    # load data
    ################################################################

    # Global Variables
    variation = 3
    var_ori = 2
    data_square = 90
    data_pentagon = 72
    data_heptagon = 51
    data_octagon = 45
    data_nonagon = 40
    data_hexagon = 60

    list_data = [data_square, data_pentagon, data_hexagon, data_heptagon, data_octagon, data_nonagon]
    list_name = ['square', 'pentagon', 'hexagon', 'heptagon', 'octagon', 'nanogan']

    all_data = []
    for (data_num, data_name) in zip(list_data, list_name):
        all_data.append(readFire(data_num, data_name, num_points=1200))

    all_data = np.concatenate(all_data, axis=0)
    np.random.seed(2023)
    np.random.shuffle(all_data)

    train_x = torch.tensor(all_data[:ntrain, :, :2], dtype=torch.float32)
    train_y = torch.tensor(all_data[:ntrain, :, 2:4], dtype=torch.float32)
    train_b = torch.tensor(all_data[:ntrain, :, 4:], dtype=torch.float32)
    valid_x = torch.tensor(all_data[-nvalid:, :, :2], dtype=torch.float32)
    valid_y = torch.tensor(all_data[-nvalid:, :, 2:4], dtype=torch.float32)
    valid_b = torch.tensor(all_data[-nvalid:, :, 4:], dtype=torch.float32)

    # x_normalizer = DataNormer(train_x.numpy(), method='mean-std', axis=(0,))
    # train_x = x_normalizer.norm(train_x)
    # valid_x = x_normalizer.norm(valid_x)
    #
    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)

    # 模型参数
    in_dim = 2
    out_dim = 2
    sampling_size = 1200
    modes = (12, 12)
    width = 32
    depth = 4
    steps = 1
    padding = 8
    dropout = 0.0

    # 训练参数
    batch_size = 32
    epochs = 1000
    learning_rate = 0.0002
    scheduler_step = 800
    scheduler_gamma = 0.1

    Logger.info('Total epochs: {:d}, learning_rate: {:e}, scheduler_step: {:d}, scheduler_gamma: {:e}'
                .format(epochs, learning_rate, scheduler_step, scheduler_gamma))

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y, train_b),
                                               batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y, valid_b),
                                               batch_size=batch_size, shuffle=False, drop_last=False)

    ################################################################
    # Neural Networks
    ################################################################

    # 建立网络
    if 'BasicPointNet' in name:
        Net_model = BasicPointNet(input_dim=in_dim, output_dim=out_dim, scaling=1.0, activation='gelu',
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
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('u', 'v'))

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

            train_coord, train_bcs, train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_coord, valid_bcs, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))

            train_coord, valid_coord = train_coord.reshape((-1, sampling_size, 2)), valid_coord.reshape(
                (-1, sampling_size, 2))
            train_true, valid_true = train_true.reshape((-1, sampling_size, out_dim)), valid_true.reshape(
                (-1, sampling_size, out_dim))
            train_pred, valid_pred = train_pred.reshape((-1, sampling_size, out_dim)), valid_pred.reshape(
                (-1, sampling_size, out_dim))

            train_true, train_pred = y_normalizer.back(train_true), y_normalizer.back(train_pred)
            valid_true, valid_pred = y_normalizer.back(valid_true), y_normalizer.back(valid_pred)

            for t in range(10):
                triang = tri.Triangulation(train_coord[t][:, 0], train_coord[t][:, 1])
                train_profile = train_coord[t][train_bcs[t, :, -1] == 0]         # 获取内部边界型线
                train_profile = ccw_sort(train_profile)    # 型线排序
                fig, axs = plt.subplots(out_dim, 3, figsize=(15, 8), num=1, layout='constrained')
                Visual.plot_fields_tr(fig, axs, train_true[t], train_pred[t], train_coord[t],
                                      triang.edges, mask=train_profile)
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(t) + '_graph.jpg'))
                plt.close(fig)

                triang = tri.Triangulation(valid_coord[t][:, 0], valid_coord[t][:, 1])
                valid_profile = valid_coord[t][valid_bcs[t, :, -1] == 0]         # 获取内部边界型线
                valid_profile = ccw_sort(valid_profile)    # 型线排序
                fig, axs = plt.subplots(out_dim, 3, figsize=(15, 8), num=1, layout='constrained')
                Visual.plot_fields_tr(fig, axs, valid_true[t], valid_pred[t], valid_coord[t],
                                      triang.edges, mask=valid_profile)
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(t) + '_graph.jpg'))
                plt.close(fig)
