#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/27 12:42
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : run_train.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Utilizes.process_data import MatLoader, DataNormer
from fno.FNOs import FNO2d
from cnn.ConvNets import UNet2d
from Utilizes.visual_data import MatplotlibVision, TextLogger

import matplotlib.pyplot as plt
import time
import os
from torchinfo import summary
import sys
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

        shape = xx.shape
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        for t in range(0, T, step):
            # y = yy[..., t:t + step]
            im = netmodel(xx.reshape(batchsize, size_x, size_y, -1), grid)[:, :, :, None, :]
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -2)
            xx = torch.cat((xx[..., step:, :], im), dim=-2)

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

            shape = xx.shape
            batchsize, size_x, size_y = shape[0], shape[1], shape[2]
            for t in range(0, T, step):
                # y = yy[..., t:t + step]
                im = netmodel(xx.reshape(batchsize, size_x, size_y, -1), grid)[:, :, :, None, :]
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -2)
                xx = torch.cat((xx[..., step:, :], im), dim=-2)

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

        shape = xx.shape
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        for t in range(0, T, step):
            # y = yy[..., t:t + step]
            im = netmodel(xx.reshape(batchsize, size_x, size_y, -1), grid)[:, :, :, None, :]
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -2)
            xx = torch.cat((xx[..., step:, :], im), dim=-2)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), grid.cpu().numpy(), yy.numpy(), pred.cpu().numpy()


def load_mat(matname):
    data = h5py.File(os.path.join('data', "data_" + matname + "_whole.mat"), mode="r")
    coords = np.array(data["grids_"], dtype=np.float32).transpose((4, 3, 2, 1, 0))[0, 0, :, :, 1:]
    fields = np.array(data["fields_"], dtype=np.float32).transpose((4, 3, 2, 1, 0))
    data = h5py.File(os.path.join('data', "dynamics_" + matname + ".mat"), mode="r")
    target = np.array(data['dynamics_200'], dtype=np.float32).transpose((2, 1, 0))
    period = np.array(data['time_period'], dtype=np.float32).transpose((1, 0))
    timeid = np.array(data['time_index'], dtype=np.int32).transpose((1, 0))

    return fields, target, coords, timeid, period


def load_all():
    fields0, target0, coords, timeid0, period0 = load_mat("6-25")
    fields1, target1, coords, timeid1, period1 = load_mat("26-35")
    fields2, target2, coords, timeid2, period2 = load_mat("36-50")

    # coords = np.concatenate((coords0, coords1, coords2), axis=0)
    fields = np.concatenate((fields0, fields1, fields2), axis=0)
    target = np.concatenate((target0, target1, target2), axis=0)
    period = np.concatenate((period0, period1, period2), axis=0)
    timeid = np.concatenate((timeid0, timeid1, timeid2), axis=0)
    # torch.save({'coords': coords, 'fields': fields, 'target': target, 'period': period, 'timeid': timeid},
    #            'data\\all.pth')
    data = h5py.File(os.path.join('data', "design_data.mat"), mode="r")
    design = np.array(data['designs'], dtype=np.float32).transpose((1, 0))
    design = np.expand_dims(design, 1).repeat(200, axis=1)

    timeno = np.mod((target[:, :, 0] - period[:, 1, np.newaxis]) / period[:, 0, np.newaxis], 1.0)
    design = np.stack((design[5:, :, 0], timeno), axis=-1)  # 前5个Re未计算

    ind = list(range(0, 25))
    return design, fields, target[:, :, 1:], coords, timeid, period


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

    design, fields, target, coords, timeid, period = load_all()

    x_normalizer = DataNormer(fields, method='mean-std')
    fields = torch.tensor(fields).permute(0, 2, 3, 1, 4)
    fields_shape = fields.shape[1:3]
    fields = x_normalizer.norm(fields)

    in_dim = 3
    out_dim = 3

    modes = (16, 8)  # fno
    steps = 4  # fno
    padding = 8  # fno
    width = 32  # all
    depth = 4  # all
    dropout = 0.0

    batch_size = 4
    epochs = 400
    learning_rate = 0.001
    scheduler_step = 300
    scheduler_gamma = 0.1

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    sub = 2
    S = 64
    T_in = steps
    T = 60
    step = 1

    ################################################################
    # load data
    ################################################################

    train_x = fields[1::2, ::sub, ::sub, :T_in]
    train_y = fields[1::2, ::sub, ::sub, T_in:T + T_in]

    valid_x = fields[0::2, ::sub, ::sub, :T_in]
    valid_y = fields[0::2, ::sub, ::sub, T_in:T + T_in]
    coords = coords[::sub, ::sub]

    del fields

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                               batch_size=batch_size, shuffle=False, drop_last=True)

    ################################################################
    # Neural Networks
    ################################################################

    # 建立网络
    if 'FNO' in name:
        Net_model = FNO2d(in_dim=in_dim, out_dim=out_dim, modes=modes, width=width, depth=depth, steps=steps,
                          padding=padding, activation='gelu').to(Device)
    elif 'UNet' in name:
        Net_model = UNet2d(in_sizes=train_x.shape[1:], out_sizes=train_y.shape[1:-1] + (out_dim,), width=width,
                           depth=depth, steps=steps, activation='gelu', dropout=dropout).to(Device)

    input1 = torch.randn(batch_size, train_x.shape[1], train_x.shape[2], train_x.shape[3], train_x.shape[4]).to(Device)
    input2 = torch.randn(batch_size, train_x.shape[1], train_x.shape[2], 2).to(Device)
    print(name)
    summary(Net_model, input_data=[input1.reshape(batch_size, train_x.shape[1], train_x.shape[2], -1), input2],
            device=Device)

    # 损失函数
    Loss_func = nn.MSELoss()
    # L1loss = nn.SmoothL1Loss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-4)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('p', 'u', 'v'))

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

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))

            for tim_id in range(0, T, 4):
                fig, axs = plt.subplots(3, 3, figsize=(18, 18), num=1)
                Visual.plot_fields_ms(fig, axs, train_true[0, ..., tim_id, :],
                                      train_pred[0, ..., tim_id, :], coords)
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(tim_id) + '_whole.jpg'))
                fig, axs = plt.subplots(3, 3, figsize=(18, 18), num=2)
                Visual.plot_fields_ms(fig, axs, train_true[0, ..., tim_id, :],
                                      train_pred[0, ..., tim_id, :], coords, cmin_max=[[-4, -5], [7, 5]])
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(tim_id) + '_local.jpg'))
                plt.close(fig)
                # Visual.plot_fields_am(fig, axs, train_true.transpose((0, 3, 1, 2))[0, ..., None],
                #                       train_pred.transpose((0, 3, 1, 2))[0, ..., None],
                #                       train_grid[0], 'train')

                fig, axs = plt.subplots(3, 3, figsize=(18, 18), num=3)
                Visual.plot_fields_ms(fig, axs, valid_true[0, ..., tim_id, :],
                                      valid_pred[0, ..., tim_id, :], coords)
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(tim_id) + '_whole.jpg'))
                fig, axs = plt.subplots(3, 3, figsize=(18, 18), num=4)
                Visual.plot_fields_ms(fig, axs, valid_true[0, ..., tim_id, :],
                                      valid_pred[0, ..., tim_id, :], coords, cmin_max=[[-4, -5], [7, 5]])
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(tim_id) + '_local.jpg'))
                plt.close(fig)
            # Visual.plot_fields_am(fig, axs, valid_true.transpose((0, 3, 1, 2))[0, ..., None],
            #                       valid_pred.transpose((0, 3, 1, 2))[0, ..., None],
            #                       valid_grid[0], 'valid')
