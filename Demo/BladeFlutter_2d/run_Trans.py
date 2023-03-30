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
from Utilizes.process_data import MatLoader, DataNormer
from transformer.Transformers_lyz import FourierTransformer2D, SimpleTransformer
from Utilizes.visual_data import MatplotlibVision, TextLogger
from Utilizes.loss_metrics import FieldsLpLoss
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
from torchinfo import summary
import sys
import yaml
import h5py


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
        # grid = grid.reshape(batch_size, 164*36,-1)
        # xx = xx.reshape(batch_size, 164*36,-1)
        # yy = yy.reshape(batch_size, 164 * 36, -1)
        pred = netmodel(xx, grid, edge=None, grid=None)['preds']
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
            # grid = grid.reshape(batch_size, 164 * 36, -1)
            # xx = xx.reshape(batch_size, 164 * 36, -1)
            # yy = yy.reshape(batch_size, 164 * 36, -1)
            pred = netmodel(xx, grid, edge=None, grid=None)['preds']
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
        # grid = grid.reshape(batch_size, 164 * 36, -1)
        # xx = xx.reshape(batch_size, 164 * 36, -1)
        # yy = yy.reshape(batch_size, 164 * 36, -1)
        pred = netmodel(xx, grid, edge=None, grid=None)['preds']
    # pred = pred.reshape(batch_size, 164, 36, -1)
    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), grid.cpu().numpy(), yy.numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'TransFFTNoGrid'
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

    in_dim = 7
    out_dim = 3
    ntrain = 200
    nvalid = 50

    modes = (12, 12)  # fno
    steps = 1  # fno
    padding = 8  # fno
    width = 32  # all
    depth = 4  # all
    dropout = 0.0

    batch_size = 12
    epochs = 3000
    learning_rate = 0.001
    scheduler_step = 2500
    scheduler_gamma = 0.5

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

    all_fields = torch.cat([F.interpolate(torch.tensor(ff[None, ...],dtype=torch.float32), [164, 36]) for ff in bld_fields], dim=0)
    fields = torch.permute(all_fields[:, :, ...], (0, 2, 3, 1))
    coords = torch.permute(all_fields[:, 1:4, ...], (0, 2, 3, 1))
    # bld_elems = [datamat[element[0]][:] for element in datamat['NEW_bld_nodes']] # 在图卷积中使用

    design1 = torch.tensor(np.transpose(datamat['boundaries'], (1, 0)).squeeze()[:, 3:5], dtype=torch.float32)
    design2 = torch.tensor(np.transpose(datamat['geometries'], (1, 0)).squeeze()[:, 2:4], dtype=torch.float32)
    design = torch.cat([design1, design2], 1)
    design = torch.tile(design[index, None, None, :], (1, coords[0].shape[0], coords[0].shape[1], 1))
    input = torch.concat([coords, design], dim=-1)
    output = fields[..., -6:-3]
    print(input.shape, output.shape)

    del datamat, coords, design, design1, design2, all_fields, bld_fields

    # train_x = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
    # train_y = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    # valid_x = input[ntrain:ntrain + nvalid, ::r1, ::r2][:, :s1, :s2]
    # valid_y = output[ntrain:ntrain + nvalid, ::r1, ::r2][:, :s1, :s2]
    train_x = input[:ntrain, ...]
    train_y = output[:ntrain, ...]
    valid_x = input[ntrain:ntrain + nvalid, ...]
    valid_y = output[ntrain:ntrain + nvalid, ...]

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
    # Neural Networks
    ################################################################

    with open(os.path.join('transformer_config.yml')) as f:
        config = yaml.full_load(f)

    config = config['BladeFlutter_2d']

    # 建立网络
    Net_model = FourierTransformer2D(**config).to(Device)
    # input1 = torch.randn(batch_size, train_x.shape[1], train_x.shape[2], train_x.shape[3]).to(Device)
    # input2 = torch.randn(batch_size, train_x.shape[1], train_x.shape[2], 2).to(Device)
    # print(name)
    # summary(Net_model, input_data=[input1, input2], device=Device)

    # 损失函数
    Loss_func = nn.MSELoss()
    Error_func = FieldsLpLoss(size_average=False)
    # L1loss = nn.SmoothL1Loss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-4)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('p', 't', 'e')) # , 'mode_x', 'mode_y', 'mode_z'

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

            for fig_id in range(10):
                fig, axs = plt.subplots(3, 3, figsize=(20, 20), layout='constrained', num=2)
                Visual.plot_fields_grid(fig, axs, train_true[fig_id], train_pred[fig_id])
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(1, 11):
                fig, axs = plt.subplots(3, 3, figsize=(20, 20), layout='constrained', num=3)
                Visual.plot_fields_grid(fig, axs, valid_true[-fig_id], valid_pred[-fig_id])
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)
                true = np.concatenate((valid_true[-fig_id, -1:, ...], valid_true[-fig_id], ), axis=0)
                pred = np.concatenate(( valid_pred[-fig_id, -1:, ...], valid_pred[-fig_id],), axis=0)
                coord = np.concatenate((valid_coord[-fig_id, -1:, ..., :3], valid_coord[-fig_id, ..., :3], ), axis=0)
                Visual.output_tecplot_struct(true, pred, coord,
                                             ['Pressure', 'Temperature', 'Static Entropy'],
                                             os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.dat'))
