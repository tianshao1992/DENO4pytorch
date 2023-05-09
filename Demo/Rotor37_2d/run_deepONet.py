#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/27 0:15
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : run_Darcy_train..py.py
"""
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn as nn
from torch.utils.data import DataLoader
from Utilizes.process_data import DataNormer
from don.DeepONets import DeepONetMulti
from Utilizes.visual_data import MatplotlibVision, TextLogger
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time
import sys
from run_MLP import get_grid, get_origin

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
    for batch, (f, x, u) in enumerate(dataloader):
        f = f.to(device)
        x = x.to(device)
        u = u.to(device)
        pred = netmodel([f, ], x, size_set=False)

        loss = lossfunc(pred, u)

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
        for batch, (f, x, u) in enumerate(dataloader):
            f = f.to(device)
            x = x.to(device)
            u = u.to(device)
            pred = netmodel([f, ], x, size_set=False)

            loss = lossfunc(pred, u)
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
        f, x, u = next(iter(dataloader))
        f = f.to(device)
        x = x.to(device)
        pred = netmodel([f, ], x, size_set=False)

    # equation = model.equation(u_var, y_var, out_pred)
    return x.cpu().numpy(), x.cpu().numpy(), u.numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'deepONet'
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

    design, fields = get_origin()

    in_dim = 28
    out_dim = 5

    ntrain = 2700
    nvalid = 200
    batch_size = 32
    batch_size2 = batch_size


    epochs = 1001
    learning_rate = 0.001
    scheduler_step = 800
    scheduler_gamma = 0.1



    print(epochs, learning_rate, scheduler_step, scheduler_gamma)
    r = out_dim*64*64
    s = 28

    ################################################################
    # load data
    ################################################################

    grid = get_grid()
    grid_trans = torch.tensor(grid[np.newaxis,:,:,:], dtype=torch.float)


    input = design
    input = torch.tensor(input, dtype=torch.float)

    output = fields
    output = torch.tensor(output, dtype=torch.float)
    print(input.shape, output.shape)

    train_f = input[:ntrain, :]
    train_u = output[:ntrain,:,:,:] #这里的u还没有展开，需要先归一化再展开
    valid_f = input[ntrain:ntrain + nvalid, :]
    valid_u = output[ntrain:ntrain + nvalid, :,:,:]
    train_grid = torch.tile(grid_trans, [train_f.shape[0], 1, 1, 1])#所有样本的坐标是一致的。
    valid_grid = torch.tile(grid_trans, [valid_f.shape[0], 1, 1, 1])

    u_show = train_u.numpy()
    gird_show = train_grid.numpy()

    f_normalizer = DataNormer(train_f.numpy(), method='mean-std')
    f_normalizer.save(os.path.join(work_path, 'x_norm.pkl'))
    train_f = f_normalizer.norm(train_f)
    valid_f = f_normalizer.norm(valid_f)

    u_normalizer = DataNormer(train_u.numpy(), method='mean-std')
    u_normalizer.save(os.path.join(work_path, 'y_norm.pkl'))
    train_u = u_normalizer.norm(train_u)
    valid_u = u_normalizer.norm(valid_u)

    # grid_normalizer = DataNormer(train_grid.numpy(), method='mean-std')#这里的axis不一样了
    # train_grid = grid_normalizer.norm(train_grid)
    # valid_grid = grid_normalizer.norm(valid_grid)

    # grid_trans = grid_trans.reshape([1, -1, 2])
    train_grid = train_grid.reshape([train_u.shape[0], -1, 2])
    valid_grid = valid_grid.reshape([valid_u.shape[0], -1, 2])
    train_u = train_u.reshape([train_u.shape[0],-1, out_dim])
    valid_u = valid_u.reshape([valid_u.shape[0],-1, out_dim])

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_f, train_grid, train_u),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_f, valid_grid, valid_u),
                                               batch_size=batch_size, shuffle=False, drop_last=True)

    ################################################################
    #  Neural Networks
    ################################################################
    # 建立网络
    Net_model = DeepONetMulti(input_dim=2, operator_dims=[28, ], output_dim=5,
                              planes_branch=[64] * 3, planes_trunk=[64] * 3).to(Device)
    # 损失函数
    Loss_func = nn.MSELoss()
    # Loss_func = nn.SmoothL1Loss()
    # L1loss = nn.SmoothL1Loss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-4)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('p', 't', 'rho', 'alf', 'v'))

    star_time = time.time()
    log_loss = [[], []]

    ################################################################
    # train process
    ################################################################

    # 生成网格文件

    for epoch in range(epochs):

        Net_model.train()
        log_loss[0].append(train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))

        Net_model.eval()
        log_loss[1].append(valid(valid_loader, Net_model, Device, Loss_func))
        print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, Optimizer.param_groups[0]['lr'], log_loss[0][-1], log_loss[1][-1], time.time() - star_time))

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
        if epoch > 0 and epoch % 100 == 0:

            # print('epoch: {:6d}, lr: {:.3e}, eqs_loss: {:.3e}, bcs_loss: {:.3e}, cost: {:.2f}'.
            #       format(epoch, learning_rate, log_loss[-1][0], log_loss[-1][1], time.time()-star_time))
            train_source, train_coord, train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_source, valid_coord, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))

            train_true = train_true.reshape([train_true.shape[0], 64, 64, out_dim])
            train_pred = train_pred.reshape([train_pred.shape[0], 64, 64, out_dim])
            valid_true = valid_true.reshape([valid_true.shape[0], 64, 64, out_dim])
            valid_pred = valid_pred.reshape([valid_pred.shape[0], 64, 64, out_dim])

            for fig_id in range(5):
                fig, axs = plt.subplots(out_dim, 3, figsize=(18, 20), num=2)
                Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], grid)
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(5):
                fig, axs = plt.subplots(out_dim, 3, figsize=(18, 20),num=3)
                Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], grid)
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

