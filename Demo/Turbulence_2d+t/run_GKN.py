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
from torch_geometric.data import Data, DataLoader
from Utilizes.process_data import MatLoader, SquareMeshGenerator
from Utilizes.loss_metrics import FieldsLpLoss
from gnn.GraphNets import KernelNN3
from Utilizes.visual_data import MatplotlibVision, TextLogger

import sklearn.metrics
import matplotlib.pyplot as plt
import time
import os
from torchinfo import summary
import sys


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
    for batch, data in enumerate(dataloader):
        data = data.to(device)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        grid = data.x[:, :2]
        xx = data.x[:, 2:]
        yy = data.y
        # grid = data.grid.to(device).tile()
        for t in range(0, T, step):
            # y = yy[..., t:t + step]
            input = torch.cat((grid, xx), dim=-1)
            im = netmodel(input, edge_index, edge_attr)
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., step:], im), dim=-1)

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
        for batch, data in enumerate(dataloader):
            data = data.to(device)
            edge_index, edge_attr = data.edge_index, data.edge_attr
            grid = data.x[:, :2]
            xx = data.x[:, 2:]
            yy = data.y
            for t in range(0, T, step):
                input = torch.cat((grid, xx), dim=-1)
                im = netmodel(input, edge_index, edge_attr)
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
                xx = torch.cat((xx[..., step:], im), dim=-1)

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
        data = next(iter(dataloader)).to(device)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        grid = data.x[:, :2]
        xx = data.x[:, 2:]
        yy = data.y
        for t in range(0, T, step):
            input = torch.cat((grid, xx), dim=-1)
            im = netmodel(input, edge_index, edge_attr)
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., step:], im), dim=-1)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), grid.cpu().numpy(), yy.cpu().numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'GKN'
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

    train_file = './data/ns_V1e-5_N1200_T20.mat'

    in_dim = 10
    out_dim = 1
    ntrain = 1000
    nvalid = 200

    # GNO
    radius = 0.05
    node_width = 16
    ker_width = 128
    depth = 4
    edge_features = 4
    node_features = 2

    batch_size = 1
    epochs = 500
    learning_rate = 0.001
    scheduler_step = 400
    scheduler_gamma = 0.1

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

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

    ################################################################
    # construct graphs
    ################################################################
    meshgenerator = SquareMeshGenerator([[0, 1], [0, 1]], [S, S])
    edge_index = meshgenerator.ball_connectivity(radius)
    edge_attr = meshgenerator.attributes(theta=None)
    grid = meshgenerator.get_grid()

    data_train = []
    for j in range(ntrain):
        # edge_attr_boundary = meshgenerator.attributes_boundary(theta=train_u[j,:])
        data_train.append(Data(torch.cat((grid, train_x[j].reshape(-1, T_in)), dim=-1),
                               y=train_y[j].reshape(-1, T), coeff=train_x[j],
                               edge_index=edge_index, edge_attr=edge_attr,
                               # edge_index_boundary=edge_index_boundary, edge_attr_boundary= edge_attr_boundary
                               ))

    data_valid = []
    for j in range(nvalid):
        data_valid.append(Data(x=torch.cat((grid, valid_x[j].reshape(-1, T_in)), dim=-1),
                               y=valid_y[j].reshape(-1, T), coeff=valid_x[j],
                               edge_index=edge_index, edge_attr=edge_attr,
                               # edge_index_boundary=edge_index_boundary, edge_attr_boundary= edge_attr_boundary
                               ))

    print(edge_index.shape, edge_attr.shape)

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(data_train, batch_size=batch_size, shuffle=False)

    ################################################################
    # Neural Networks
    ################################################################

    # 建立网络
    # Net_model = GMMNet(in_dim=2, out_dim=1, edge_dim=4, width=width, depth=depth, activation='gelu').to(device)
    Net_model = KernelNN3(width_node=node_width, width_kernel=ker_width,
                          depth=4, ker_in=4, in_width=12, out_width=1).to(Device)
    # 损失函数
    Loss_func = nn.MSELoss()
    # Loss_func = FieldsLpLoss(size_average=False)
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
    data = next(iter(train_loader))
    data = data.to(Device)
    edge_index, edge_attr = data.edge_index, data.edge_attr
    summary(Net_model, input_data=[data.x, edge_index, edge_attr], device=Device)

    for epoch in range(epochs):

        Net_model.train()
        log_loss[0].append(train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))

        Net_model.eval()
        log_loss[1].append(valid(valid_loader, Net_model, Device, Loss_func))
        print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, Optimizer.param_groups[0]['lr'], log_loss[0][-1], log_loss[1][-1], time.time() - star_time))

        star_time = time.time()

        if epoch > 0 and epoch % 5 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(work_path, 'log_loss.svg'))
            plt.close(fig)

        ################################################################
        # Visualization
        ################################################################

        if epoch >= 0 and epoch % 50 == 0:
            # print('epoch: {:6d}, lr: {:.3e}, eqs_loss: {:.3e}, bcs_loss: {:.3e}, cost: {:.2f}'.
            #       format(epoch, learning_rate, log_loss[-1][0], log_loss[-1][1], time.time()-star_time))
            train_coord, train_grid, train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_coord, valid_grid, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

            train_grid = train_grid.reshape(-1, 64, 64, 2)
            train_true = train_true.reshape(-1, 64, 64, T)
            train_pred = train_pred.reshape(-1, 64, 64, T)

            valid_grid = valid_grid.reshape(-1, 64, 64, 2)
            valid_true = valid_true.reshape(-1, 64, 64, T)
            valid_pred = valid_pred.reshape(-1, 64, 64, T)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))

            for tim_id in range(0, T, 1):
                fig, axs = plt.subplots(1, 3, figsize=(18, 5), num=1)
                Visual.plot_fields_ms(fig, axs, train_true[0, ..., tim_id, None],
                                      train_pred[0, ..., tim_id, None], train_grid[0])

                fig.savefig(os.path.join(work_path, 'train_solution_' + str(tim_id) + '.jpg'))
                plt.close(fig)
                # Visual.plot_fields_am(fig, axs, train_true.transpose((0, 3, 1, 2))[0, ..., None],
                #                       train_pred.transpose((0, 3, 1, 2))[0, ..., None],
                #                       train_grid[0], 'train')

                fig, axs = plt.subplots(1, 3, figsize=(18, 5), num=2)
                Visual.plot_fields_ms(fig, axs, valid_true[0, ..., tim_id, None],
                                      valid_pred[0, ..., tim_id, None], valid_grid[0])
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(tim_id) + '.jpg'))
                plt.close(fig)
            # Visual.plot_fields_am(fig, axs, valid_true.transpose((0, 3, 1, 2))[0, ..., None],
            #                       valid_pred.transpose((0, 3, 1, 2))[0, ..., None],
            #                       valid_grid[0], 'valid')
