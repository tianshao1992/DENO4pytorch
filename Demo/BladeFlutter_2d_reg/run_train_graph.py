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
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from gnn.GraphNets import KernelNN3, GMMNet
from Utilizes.loss_metrics import FieldsLpLoss
from Utilizes.visual_data import MatplotlibVision, TextLogger
from Utilizes.process_data import DataNormer
from Utilizes.loss_metrics import FieldsLpLoss
import matplotlib.pyplot as plt
import time
import os
import h5py
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
    width = 64
    depth = 4
    name = 'GMM_Net_' + str(width) + '_' + str(depth) #'GCN_KernelNN3'
    work_path = os.path.join('work', name + 'ForPTE')
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    sys.stdout = TextLogger(os.path.join(work_path, 'train.log'), sys.stdout)
    print(work_path)

    in_dim = 7
    out_dim = 3
    ntrain = 200
    nvalid = 50


    steps = 1

    batch_size = 16
    epochs = 3000
    learning_rate = 0.01
    scheduler_step = int(epochs*0.7)
    scheduler_gamma = 0.1

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    ################################################################
    # load data
    ################################################################
    data_file = os.path.join('data', "flutter_bld_res1_1-300.mat")

    datamat = h5py.File(data_file)
    bld_fields= []
    index = []
    for ind, element in enumerate(datamat['NEW_bld_fields']):
        if np.size(datamat[element[0]][:]) > 10:
            bld_fields.append(datamat[element[0]][:])
            index.append(ind)

    all_fields = torch.cat([F.interpolate(torch.tensor(ff[None, ...], dtype=torch.float32), [164, 36]) for ff in bld_fields], dim=0)
    fields = torch.permute(all_fields[:, -6:-3, ...], (0, 2, 3, 1))
    coords = torch.permute(all_fields[:, 1:4, ...], (0, 2, 3, 1))
    # bld_elems = [datamat[element[0]][:] for element in datamat['NEW_bld_nodes']] # 在图卷积中使用

    design1 = torch.tensor(np.transpose(datamat['boundaries'], (1, 0)).squeeze()[:, 3:5], dtype=torch.float32)
    design2 = torch.tensor(np.transpose(datamat['geometries'], (1, 0)).squeeze()[:, 2:4], dtype=torch.float32)
    design = torch.cat([design1, design2], 1)
    design = torch.tile(design[index, None, None, :], (1, coords[0].shape[0], coords[0].shape[1], 1))
    input = torch.concat([coords, design], dim=-1)
    del datamat, coords, design, design1, design2, all_fields, bld_fields

    x_normalizer = DataNormer(input.numpy()[:ntrain], method='mean-std')
    input = x_normalizer.norm(input)
    y_normalizer = DataNormer(fields.numpy()[:ntrain], method='mean-std')
    fields = y_normalizer.norm(fields)

    edge_index = []
    for i in range(164):
        for j in range(36):
            if j - 1 >= 0:
                edge_index.append([i * 36 + j, i * 36 + j - 1])
            if j + 1 <= 35:
                edge_index.append([i * 36 + j, i * 36 + j + 1])
            if i - 1 >= 0:
                edge_index.append([i * 36 + j, i * 36 + j - 36])
            if i + 1 <= 163:
                edge_index.append([i * 36 + j, i * 36 + j + 36])
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    data_train = []
    for i in range(ntrain):
        x = input[i].reshape((-1, 7))
        y = fields[i].reshape((-1, 3))
        edge_attr = torch.norm(x[edge_index][:, 0, :3] - x[edge_index][:, 1, :3], dim=-1, keepdim=True)
        data_train.append(Data(x=x, y=y, edge_index=edge_index.T, edge_attr=edge_attr))

    data_valid = []
    for i in range(nvalid):
        x = input[ntrain:][i].reshape((-1, 7))
        y = fields[ntrain:][i].reshape((-1, 3))
        edge_attr = torch.norm(x[edge_index][:, 0, :3] - x[edge_index][:, 1, :3], dim=-1, keepdim=True)
        data_valid.append(Data(x=x, y=y, edge_index=edge_index.T, edge_attr=edge_attr))

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=False)

    ################################################################
    # Neural Networks
    ################################################################

    # 建立网络

    Net_model = GMMNet(in_dim=in_dim, out_dim=out_dim, edge_dim=1, width=width, depth=depth, activation='gelu').to(device)
    # Net_model = KernelNN3(16, 32, depth, 1, in_width=in_dim, out_width=out_dim).to(device)

    # 损失函数
    Loss_func = nn.MSELoss()
    Error_func = FieldsLpLoss(size_average=False)
    # L1loss = nn.SmoothL1Loss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9))
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('p', 't', 'e'))

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
            train_coord, train_true, train_pred = inference(train_loader, Net_model, device)
            valid_coord, valid_true, valid_pred = inference(valid_loader, Net_model, device)

            Error_func.p = 1
            ErrL1a = Error_func.abs(valid_pred, valid_true)
            ErrL1r = Error_func.rel(valid_pred, valid_true)
            Error_func.p = 2
            ErrL2a = Error_func.abs(valid_pred, valid_true)
            ErrL2r = Error_func.rel(valid_pred, valid_true)

            fig, axs = plt.subplots(1, 2, figsize=(10, 10), layout='constrained', num=3)
            Visual.plot_box(fig, axs[0], ErrL1r, legends=Visual.field_name)
            Visual.plot_box(fig, axs[1], ErrL2r, legends=Visual.field_name)
            fig.savefig(os.path.join(work_path, 'valid_box.jpg'))
            plt.close(fig)

            train_coord = x_normalizer.back(train_coord)
            valid_coord = x_normalizer.back(valid_coord)
            train_true, valid_true = y_normalizer.back(train_true), y_normalizer.back(valid_true)
            train_pred, valid_pred = y_normalizer.back(train_pred), y_normalizer.back(valid_pred)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict(),
                        'valid_coord': valid_coord, 'valid_true': valid_true, 'valid_pred': valid_pred,
                        'ErrL1a': ErrL1a, 'ErrL1r': ErrL1r, 'ErrL2a': ErrL2a, 'ErrL2r': ErrL2r,
                        },
                       os.path.join(work_path, 'latest_model.pth'))

            train_coord = train_coord.reshape((batch_size, 164, 36, -1))
            train_true = train_true.reshape((batch_size, 164, 36, -1))
            train_pred = train_pred.reshape((batch_size, 164, 36, -1))
            valid_coord = valid_coord.reshape((batch_size, 164, 36, -1))
            valid_true = valid_true.reshape((batch_size, 164, 36, -1))
            valid_pred = valid_pred.reshape((batch_size, 164, 36, -1))

            for fig_id in range(10):
                fig, axs = plt.subplots(3, 3, figsize=(20, 20), layout='constrained', num=2)
                Visual.plot_fields_grid(fig, axs, train_true[fig_id], train_pred[fig_id])
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(10):
                fig, axs = plt.subplots(3, 3, figsize=(20, 20), layout='constrained', num=3)
                Visual.plot_fields_grid(fig, axs, valid_true[fig_id], valid_pred[fig_id])
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)
                Visual.output_tecplot_struct(valid_true, valid_pred,valid_coord,
                                             ['Pressure', 'Temperature', 'Static Entropy'],
                                             os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.dat'))

