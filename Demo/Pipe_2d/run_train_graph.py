#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/12/4 1:51
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : run_train_graph.py
"""

import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from gnn.GraphNets import KernelNN3
from Utilizes.loss_metrics import FieldsLpLoss
from Utilizes.visual_data import MatplotlibVision

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
    batch_size = dataloader.batch_size
    for batch, data in enumerate(dataloader):
        data = data.to(device)
        pred = netmodel(data)
        loss = lossfunc(pred.view(batch_size, -1), data.y.view(batch_size, -1))

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
    batch_size = dataloader.batch_size
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            data = data.to(device)
            pred = netmodel(data)
            loss = lossfunc(pred.view(batch_size, -1), data.y.view(batch_size, -1))
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
        data = next(iter(dataloader))
        data = data.to(device)
        pred = netmodel(data)

    return data.x.cpu().numpy(), data.y.cpu().numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'NS-2d-Pipe'
    work_path = os.path.join('work')
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    INPUT_X = './data/Pipe_X.npy'
    INPUT_Y = './data/Pipe_Y.npy'
    OUTPUT_Sigma = './data/Pipe_Q.npy'

    in_dim = 2
    out_dim = 1
    ntrain = 1000
    nvalid = 200

    width = 32
    depth = 4
    steps = 1

    batch_size = 10
    epochs = 500
    learning_rate = 0.001
    scheduler_step = 400
    scheduler_gamma = 0.1

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    r1 = 1
    r2 = 1
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
    print(input.shape, output.shape)

    train_x = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
    train_y = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    valid_x = input[ntrain:ntrain + nvalid, ::r1, ::r2][:, :s1, :s2]
    valid_y = output[ntrain:ntrain + nvalid, ::r1, ::r2][:, :s1, :s2]

    edge_index = []
    for i in range(129):
        for j in range(129):
            if j - 1 >= 0:
                edge_index.append([i * 129 + j, i * 129 + j - 1])
            if j + 1 <= 128:
                edge_index.append([i * 129 + j, i * 129 + j + 1])
            if i - 1 >= 0:
                edge_index.append([i * 129 + j, i * 129 + j - 129])
            if i + 1 <= 128:
                edge_index.append([i * 129 + j, i * 129 + j + 129])
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    data_train = []
    for i in range(ntrain):
        x = train_x[i].reshape((129 * 129, -1))
        y = train_y[i].reshape((129 * 129, -1))
        edge_attr = torch.norm(x[edge_index][:, 0] - x[edge_index][:, 1], dim=-1, keepdim=True)
        data_train.append(Data(x=x, y=y, edge_index=edge_index.T, edge_attr=edge_attr))

    data_valid = []
    for i in range(nvalid):
        x = valid_x[i].reshape((129 * 129, -1))
        y = valid_y[i].reshape((129 * 129, -1))
        edge_attr = torch.norm(x[edge_index][:, 0] - x[edge_index][:, 1], dim=-1, keepdim=True)
        data_valid.append(Data(x=x, y=y, edge_index=edge_index.T, edge_attr=edge_attr))

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=False)

    ################################################################
    # Neural Networks
    ################################################################

    # 建立网络

    # Net_model = GMMNet(in_dim=in_dim, out_dim=out_dim, edge_dim=1, width=width, depth=depth, activation='gelu').to(device)
    Net_model = KernelNN3(16, 32, depth, 1, in_width=in_dim, out_width=1).to(device)

    # 损失函数
    # Loss_func = nn.MSELoss()
    Loss_func = FieldsLpLoss(size_average=False)
    # L1loss = nn.SmoothL1Loss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9))
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
        log_loss[0].append(train(train_loader, Net_model, device, Loss_func, Optimizer, Scheduler))

        Net_model.eval()
        log_loss[1].append(valid(valid_loader, Net_model, device, Loss_func))
        print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, learning_rate, log_loss[0][-1], log_loss[1][-1], time.time() - star_time))

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

        if epoch > 0 and epoch % 20 == 0:
            # print('epoch: {:6d}, lr: {:.3e}, eqs_loss: {:.3e}, bcs_loss: {:.3e}, cost: {:.2f}'.
            #       format(epoch, learning_rate, log_loss[-1][0], log_loss[-1][1], time.time()-star_time))
            train_coord, train_true, train_pred = inference(train_loader, Net_model, device)
            valid_coord, valid_true, valid_pred = inference(valid_loader, Net_model, device)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))

            train_coord = train_coord.reshape((batch_size, 129, 129, -1))
            train_coord[..., 0] *= 10.
            train_true = train_true.reshape((batch_size, 129, 129, -1))
            train_pred = train_pred.reshape((batch_size, 129, 129, -1))
            valid_coord = valid_coord.reshape((batch_size, 129, 129, -1))
            valid_coord[..., 0] *= 10.
            valid_true = valid_true.reshape((batch_size, 129, 129, -1))
            valid_pred = valid_pred.reshape((batch_size, 129, 129, -1))

            for fig_id in range(10):
                fig, axs = plt.subplots(1, 3, figsize=(18, 6), layout='constrained', num=2)
                Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], train_coord[fig_id])
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(10):
                fig, axs = plt.subplots(1, 3, figsize=(18, 6), layout='constrained', num=3)
                Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], valid_coord[fig_id])
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)
