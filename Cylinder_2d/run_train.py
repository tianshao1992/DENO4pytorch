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
from Utilizes.process_data import DataNormer, MatLoader
from Models.FNOs import FNO2d
from Models.ConvNets import UNet2d, UpSampleNet2d, DownSampleNet2d
from Utilizes.loss_metrics import FieldsLpLoss
from Utilizes.visual_data import MatplotlibVision, TextLogger

import matplotlib.pyplot as plt
import time
import os
from torchinfo import summary
import sys
import h5py
from scipy import io as sio


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
        if 'fno' in modelType:
            gd = feature_transform(xx)
            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = netmodel(xx, gd)
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
                xx = torch.cat((xx[..., step:], im), dim=-1)
        elif 'unet' in modelType:
            pred = netmodel(xx)

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
            if 'fno' in modelType:
                gd = feature_transform(xx)
                for t in range(0, T, step):
                    y = yy[..., t:t + step]
                    im = netmodel(xx, gd)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                    xx = torch.cat((xx[..., step:], im), dim=-1)
            elif 'unet' in modelType:
                pred = netmodel(xx)
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
        if 'fno' in modelType:
            gd = feature_transform(xx)
            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = netmodel(xx, gd)
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
                xx = torch.cat((xx[..., step:], im), dim=-1)
        elif 'unet' in modelType:
            pred = netmodel(xx)
            gd = torch.zeros((1,)).to(device)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), gd.cpu().numpy(), yy.numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'Euler-2d-NACA'
    work_path = os.path.join('work')
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    # ??????????????????????????????a.log?????????????????????a.txt
    sys.stdout = TextLogger(os.path.join(work_path, 'train.log'), sys.stdout)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_file = './data/cyl_Re250.mat'

    in_dim = 10
    out_dim = 1
    ntrain = 150
    nvalid = 50

    modelType = 'fno'
    modes = (12, 12)  # fno
    steps = 1  # fno
    padding = 8  # fno
    width = 32  # all
    depth = 4  # all
    dropout = 0.0

    batch_size = 16
    epochs = 500
    learning_rate = 0.001
    scheduler_step = 400
    scheduler_gamma = 0.1

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    sub = 1
    S = 64
    T_in = 10
    T = 40
    step = 1

    ################################################################
    # load data
    ################################################################

    reader = MatLoader(train_file)
    train_x = reader.read_field('fields_').squeeze(0)[:ntrain, ::sub, ::sub, :T_in]
    train_y = reader.read_field('u').squeeze(0)[:ntrain, ::sub, ::sub, T_in:T + T_in]

    valid_x = reader.read_field('u')[ntrain:, ::sub, ::sub, :T_in]
    valid_y = reader.read_field('u')[ntrain:, ::sub, ::sub, T_in:T + T_in]
    del reader

    # x_normalizer = DataNormer(train_x.numpy(), method='mean-std', axis=(0,))
    # train_x = x_normalizer.norm(train_x)
    # valid_x = x_normalizer.norm(valid_x)
    #
    # y_normalizer = DataNormer(train_y.numpy(), method='mean-std', axis=(0,))
    # train_y = y_normalizer.norm(train_y)
    # valid_y = y_normalizer.norm(valid_y)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                               batch_size=batch_size, shuffle=False, drop_last=True)

    ################################################################
    # Neural Networks
    ################################################################

    # ????????????
    if modelType == 'fno':
        Net_model = FNO2d(in_dim=in_dim, out_dim=out_dim, modes=modes, width=width, depth=depth, steps=steps,
                          padding=padding, activation='gelu').to(device)
        input1 = torch.randn(batch_size, train_x.shape[1], train_x.shape[2], train_x.shape[3]).to(device)
        input2 = torch.randn(batch_size, train_x.shape[1], train_x.shape[2], 2).to(device)
        print(modelType)
        summary(Net_model, input_data=[input1, input2], device=device)
    elif modelType == 'unet':
        Net_model = UNet2d(in_sizes=train_x.shape[1:], out_sizes=train_y.shape[1:], width=width, depth=depth,
                           activation='gelu', dropout=dropout).to(device)
        input1 = torch.randn(batch_size, train_x.shape[1], train_x.shape[2], train_x.shape[3]).to(device)
        print(modelType)
        summary(Net_model, input_data=input1, device=device)

    # ????????????
    # Loss_func = nn.MSELoss()
    Loss_func = FieldsLpLoss()
    # L1loss = nn.SmoothL1Loss()
    # ????????????
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-4)
    # ????????????
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # ?????????
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
            train_coord, train_grid, train_true, train_pred = inference(train_loader, Net_model, device)
            valid_coord, train_grid, valid_true, valid_pred = inference(valid_loader, Net_model, device)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))

            for fig_id in range(10):
                fig, axs = plt.subplots(2, 3, figsize=(18, 10), layout='constrained', num=2)
                Visual.plot_fields_ms(fig, axs[0], train_true[fig_id], train_pred[fig_id], train_coord[fig_id])
                Visual.plot_fields_ms(fig, axs[1], train_true[fig_id, nx:-nx, :ny],
                                      train_pred[fig_id, nx:-nx, :ny], train_coord[fig_id, nx:-nx, :ny])
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(10):
                fig, axs = plt.subplots(2, 3, figsize=(18, 10), layout='constrained', num=3)
                Visual.plot_fields_ms(fig, axs[0], valid_true[fig_id], valid_pred[fig_id], valid_coord[fig_id])
                Visual.plot_fields_ms(fig, axs[1], valid_true[fig_id, nx:-nx, :ny],
                                      valid_pred[fig_id, nx:-nx, :ny], valid_coord[fig_id, nx:-nx, :ny])
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)
