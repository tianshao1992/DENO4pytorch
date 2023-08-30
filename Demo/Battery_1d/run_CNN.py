#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/1/29 19:47
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : run_train.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Utilizes.process_data import DataNormer, MatLoader
from fno.FNOs import FNO1d
from cnn.ConvNets import DownSampleNet1d
from basic.basic_layers import FcnSingle
from Utilizes.visual_data import MatplotlibVision, TextLogger
from torchinfo import summary

import matplotlib.pyplot as plt
import time
import os
import sys

import warnings
warnings.filterwarnings("ignore")

class predictor(nn.Module):

    def __init__(self, branch, trunk, out_dim):

        super(predictor, self).__init__()

        self.branch_net = branch
        self.trunc_net = trunk
        self.target_net = nn.Linear(branch.planes[-1], out_dim)


    def forward(self, input, conv):
        """
        forward compute
        :param input: tensor list[(batch_size, ..., operator_dims[0]), (batch_size, ..., operator_dims[1]), ...]
        :param conv: (batch_size, ..., input_dim)
        """

        T = self.trunc_net(input)
        B = self.branch_net(conv)
        feature = B * T
        F = self.target_net(feature)
        return F

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
    for batch, (xx, ww, yy) in enumerate(dataloader):
        xx = xx.to(device)
        ww = ww.to(device)
        yy = yy.to(device)

        pred = netmodel(xx, ww)
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
        for batch, (xx, ww, yy) in enumerate(dataloader):
            xx = xx.to(device)
            ww = ww.to(device)
            yy = yy.to(device)
            pred = netmodel(xx, ww)
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

    true = []
    pred = []
    with torch.no_grad():
        for batch, (xx, ww, yy) in enumerate(dataloader):
            xx = xx.to(device)
            ww = ww.to(device)
            pred.append(netmodel(xx, ww).cpu().numpy())
            true.append(yy.cpu().numpy())

    # equation = model.equation(u_var, y_var, out_pred)
    return np.concatenate(true, axis=0), np.concatenate(pred, axis=0)


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'CNN'
    work_path = os.path.join('work', name)
    train_path = os.path.join(work_path)
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    # 将控制台的结果输出到log文件
    Logger = TextLogger(os.path.join(work_path, 'train.log'))

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    Logger.info("Model Name: {:s}, Computing Device: {:s}".format(name, str(Device)))

    ntrain = 4000
    nvalid = 1000

    modes = 16
    width = 32
    depth = 6
    steps = 1
    padding = 2
    dropout = 0.0
    batch_size = 128

    epochs = 300
    learning_rate = 0.001
    scheduler_step = 250
    scheduler_gamma = 0.1

    Logger.info('Total epochs: {:d}, learning_rate: {:e}, scheduler_step: {:d}, scheduler_gamma: {:e}'
                .format(epochs, learning_rate, scheduler_step, scheduler_gamma))

    sub = 1  # subsampling rate
    h = 2 ** 13 // sub  # total grid size divided by the subsampling rate
    s = h

    ################################################################
    # load data
    ################################################################

    signal = torch.tensor(np.load('data/data.npy').transpose((0, 2, 1)), dtype=torch.float32)
    target = torch.tensor(np.load('data/label.npy')[:, None].astype(np.float32), dtype=torch.float32)
    weight = torch.tensor(np.load('data/weight.npy')[:, None].astype(np.float32), dtype=torch.float32)

    ind = target[:, 0] > 200
    signal = signal[ind]
    target = target[ind]
    weight = weight[ind]

    signal_normer = DataNormer(signal)
    target_normer = DataNormer(target)
    weight_normer = DataNormer(weight)

    train_x = signal_normer.norm(signal[:ntrain, ::sub])
    train_w = weight_normer.norm(weight[:ntrain])
    train_y = target_normer.norm(target[:ntrain])

    valid_x = signal_normer.norm(signal[ntrain:ntrain+nvalid, ::sub])
    valid_w = weight_normer.norm(weight[ntrain:ntrain+nvalid])
    valid_y = target_normer.norm(target[ntrain:ntrain+nvalid])

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_w, train_y),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_w, valid_y),
                                               batch_size=batch_size, shuffle=False, drop_last=True)

    ################################################################
    #  Neural Networks
    ################################################################

    # 建立网络

    trunk_net = DownSampleNet1d(in_sizes=train_x.shape[1:], out_sizes=64, width=width,
                                depth=depth, activation='gelu', dropout=dropout).to(Device)

    branch_net = FcnSingle(planes=(train_w.shape[-1], 64, 64), last_activation=True).to(Device)

    Net_model = predictor(trunk=trunk_net, branch=branch_net, out_dim=train_y.shape[-1]).to(Device)

    input1 = torch.randn(batch_size, train_x.shape[1], train_x.shape[2]).to(Device)
    input2 = torch.randn(batch_size, train_w.shape[1]).to(Device)
    model_statistics = summary(Net_model, input_data=[input1, input2], device=Device, verbose=0)
    Logger.write(str(model_statistics))

    # 损失函数
    Loss_func = nn.MSELoss()
    # L1loss = nn.SmoothL1Loss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-5)
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
        Logger.info('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, Optimizer.param_groups[0]['lr'], log_loss[0][-1], log_loss[1][-1], time.time() - star_time))

        star_time = time.time()

        if epoch > 0 and epoch % 20 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.array(log_loss)[0, :], 'train_step')
            Visual.plot_loss(fig, axs, np.array(log_loss)[1, :], 'valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(work_path, 'log_loss.svg'))
            plt.close(fig)

        ################################################################
        # Visualization
        ################################################################

        if epoch > 0 and epoch % 20 == 0:
            # print('epoch: {:6d}, lr: {:.3e}, eqs_loss: {:.3e}, bcs_loss: {:.3e}, cost: {:.2f}'.
            #       format(epoch, learning_rate, log_loss[-1][0], log_loss[-1][1], time.time()-star_time))
            train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_true, valid_pred = inference(valid_loader, Net_model, Device)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))

            train_true = target_normer.back(train_true)
            train_pred = target_normer.back(train_pred)
            valid_true = target_normer.back(valid_true)
            valid_pred = target_normer.back(valid_pred)

            train_error = (train_pred - train_true) / train_true
            valid_error = (valid_pred - valid_true) / valid_true

            Logger.info('mean train error: {:.3e}, mean valid error: {:.3e}'
                        .format(np.mean(np.abs(train_error)), np.mean(np.abs(valid_error))))

            fig, axs = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)

            Visual.plot_regression(fig, axs[0, 0], train_true[:, 0], train_pred[:, 0], error_ratio=0.01, )
            Visual.plot_error(fig, axs[1, 0], train_error[:, 0], error_ratio=0.003, rel_error=True)
            Visual.plot_regression(fig, axs[0, 1], valid_true[:, 0], valid_pred[:, 0], error_ratio=0.01, )
            Visual.plot_error(fig, axs[1, 1], valid_error[:, 0], error_ratio=0.003, rel_error=True)

            fig.savefig(os.path.join(work_path, 'pred_results.jpg'))
            plt.close(fig)
