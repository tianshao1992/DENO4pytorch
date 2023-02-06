#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/2/4 0:15
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : run_deepONet+PINN.py.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Utilizes.process_data import MatLoader
from basic_layers import DeepONetMulti
from pinn.differ_layers import gradients
from Utilizes.loss_metrics import FieldsLpLoss
from Utilizes.visual_data import MatplotlibVision, TextLogger

import matplotlib.pyplot as plt
import time
import os
import sys


class Net(DeepONetMulti):
    """use basic model to build the network"""

    def __init__(self, input_dim, operator_dims, output_dim, planes_branch, planes_trunk):
        super(Net, self).__init__(input_dim, operator_dims, output_dim, planes_branch, planes_trunk)

    def equation(self, u_var, y_var, out_var, ux):
        """
        Args:
            u_var: branch input
            y_var: trunk input
            out_var: predicted fields
            ux: input function value of each node
        Returns:
            res: equation residual
        """
        dfda = gradients(out_var, y_var)
        dfdx, dfdt = dfda[..., (0,)], dfda[..., (1,)]
        res = dfdt + ux * dfdx

        return res


def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler, bcs_index):
    """
    Args:
        data_loader: output fields at last time step
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """

    train_loss = 0
    train_loss_eqs = 0
    train_loss_bcs = 0
    for batch, (f, x, u) in enumerate(dataloader):
        f = f.to(device)
        x = x.to(device)
        u = u.to(device)
        x.requires_grad_(True)
        optimizer.zero_grad()
        pred = netmodel([f, ], x, size_set=False)
        resi = netmodel.equation(f, x, pred, u)

        # 守恒残差
        eqs_loss = torch.mean(resi[:, bcs_index[0]:] ** 2)
        # 边界条件
        bcs_loss = torch.mean((pred[:, :bcs_index[0]] - u[:, :bcs_index[0]]) ** 2)

        loss = eqs_loss + bcs_loss * 10.

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loss_eqs += eqs_loss.item()
        train_loss_bcs += bcs_loss.item()

    scheduler.step()
    return train_loss / (batch + 1), train_loss_eqs / (batch + 1), train_loss_bcs / (batch + 1)


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
        # u = u.to(device)
        # u = u.to(device)
        pred = netmodel([f, ], x, size_set=False)

    # equation = model.equation(u_var, y_var, out_pred)
    return f.cpu().numpy(), x.cpu().numpy(), u.numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'deepONet-PINN'
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

    train_file = os.path.join('data', 'pinn_train.mat')
    valid_file = os.path.join('data', 'pinn_valid.mat')

    in_dim = 1
    out_dim = 1
    ntrain = 2000
    nvalid = 200
    batch_size = 32

    epochs = 1000
    learning_rate = 0.001
    scheduler_step = 800
    scheduler_gamma = 0.1

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    ################################################################
    # load data
    ################################################################

    reader = MatLoader(train_file)
    train_f = reader.read_field('u_bcs_train')[:ntrain, 0]
    # 注意: 本算例中训练集的s_res_train 不是真实解，而是每个节点对应的branch_net输入函数值，用于计算equation_loss
    train_u = torch.cat((reader.read_field('s_bcs_train'), reader.read_field('s_res_train')), dim=1)[:ntrain]
    train_grid = torch.cat((reader.read_field('y_bcs_train'),  reader.read_field('y_res_train')), dim=1)[:ntrain]

    reader = MatLoader(valid_file)
    valid_f = reader.read_field('u_res_valid')[:, 0]
    valid_u = reader.read_field('s_res_valid')[..., None]
    valid_grid = reader.read_field('y_res_valid')

    del reader

    # f_normalizer = DataNormer(raw_data.reshape(raw_data.shape[0], -1), method='mean-std', axis=(0,))
    # train_f = f_normalizer.norm(train_f)
    # valid_f = f_normalizer.norm(valid_f)

    operator_dim = train_f.shape[-1]

    # grid_normalizer = DataNormer(train_grid, method='mean-std', axis=(0, 1))
    # train_grid = grid_normalizer.norm(train_grid)
    # valid_grid = grid_normalizer.norm(valid_grid)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_f, train_grid, train_u),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_f, valid_grid, valid_u),
                                               batch_size=batch_size, shuffle=False, drop_last=True)

    ################################################################
    #  Neural Networks
    ################################################################
    # 建立网络
    Net_model = Net(input_dim=2, operator_dims=[operator_dim, ], output_dim=1,
                    planes_branch=[64] * 6, planes_trunk=[64] * 6).to(Device)
    # 损失函数
    Loss_func = nn.MSELoss()
    # L1loss = nn.SmoothL1Loss()
    # 评价指标
    Field_metric = FieldsLpLoss(d=2, p=2, reduction=True, size_average=False)
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-4)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('f',))

    star_time = time.time()
    log_loss = [[], [], []]

    ################################################################
    # train process
    ################################################################

    # 生成网格文件

    for epoch in range(epochs):

        Net_model.train()
        loss_tol, loss_eqs, loss_bcs = train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler, [200, ])
        log_loss[0].append(loss_tol)
        log_loss[1].append(loss_eqs)
        log_loss[2].append(loss_bcs)

        print('epoch: {:6d}, lr: {:.3e}, tol_loss: {:.3e}, eqs_loss: {:.3e}, bcs_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, Optimizer.param_groups[0]['lr'], log_loss[0][-1], log_loss[1][-1], log_loss[2][-1],
                     time.time() - star_time))

        star_time = time.time()

        if epoch > 0 and epoch % 50 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'tol_loss')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'eqs_loss')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[2, :], 'bcs_loss')
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

            err_rel = Field_metric.rel(valid_pred, valid_true)

            for fig_id in range(5):
                # 100 为验证集算例中的网格节点个数
                fig, axs = plt.subplots(1, 3, figsize=(18, 5), layout='constrained', num=3)
                Visual.plot_fields_ms(fig, axs, valid_true[fig_id].reshape((100, 100, -1)),
                                      valid_pred[fig_id].reshape((100, 100, -1)),
                                      valid_coord[fig_id].reshape((100, 100, -1)))
                plt.title('Absolute error:  {:.3e}'.format(float(err_rel[fig_id])), fontdict=Visual.font_EN)
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)