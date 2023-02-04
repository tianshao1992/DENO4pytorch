#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/27 0:15
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : run_Darcy_train..py.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Utilizes.process_data import DataNormer, MatLoader
from Models.basic_layers import DeepONetMulti
from Models.differ_layers import gradients
from Utilizes.loss_metrics import FieldsLpLoss
from Utilizes.visual_data import MatplotlibVision, TextLogger

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import time
import os
import sys


class Net(DeepONetMulti):
    """use basic model to build the network"""

    def __init__(self, input_dim, operator_dims, output_dim, planes_branch, planes_trunk):
        super(Net, self).__init__(input_dim, operator_dims, output_dim, planes_branch, planes_trunk)

    def equation(self, u_var, y_var, out_var):
        """
        Args:
            u_var: branch input
            y_var: trunk input
            out_var: predicted fields
        Returns:
            res: equation residual
        """
        dfda = gradients(out_var, y_var)
        dfdx, dfdy = dfda[..., (0,)], dfda[..., (1,)]
        res = dfdx ** 2 + dfdy ** 2

        return res


def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler, sizes):
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
        resi = netmodel.equation(f, x, pred)

        # 守恒残差
        eqs_loss = torch.mean((resi[:, sizes[0]:] - 1.0) ** 2)
        # 边界条件
        bcs_loss = torch.mean(pred[:, :sizes[0]] ** 2)

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


def gen_data(all_data, mode, star, size):
    dom_coords = np.array([[-3, -3],
                           [3, 3]], dtype=np.float32)

    Num_bcs = all_data.shape[1]
    Num_eqs = 500
    Num_grid = 101
    xx = np.linspace(dom_coords[0, 0], dom_coords[1, 0], Num_grid, dtype=np.float32)
    yy = np.linspace(dom_coords[0, 1], dom_coords[1, 1], Num_grid, dtype=np.float32)
    xx, yy = np.meshgrid(xx, yy)

    D = all_data.reshape(-1, 2)
    mu = D.mean(0)
    sigma = D.std(0)

    u_r_list = []
    y_r_list = []
    s_r_list = []
    for k in range(size):
        # Create training data for bcs loss
        u = (all_data[star + k] - mu) / sigma

        if mode == "train":
            y_eqs = dom_coords.min(axis=0) + (dom_coords.max(axis=0) - dom_coords.min(axis=0)) \
                    * np.random.uniform(size=(Num_eqs, 2)).astype(np.float32)  # shape = (Num_eqs, 2)

            u_r_list.append(u.reshape(-1))
            y_r_list.append(np.concatenate((u, y_eqs), axis=0))
            s_r_list.append(np.zeros((Num_bcs + Num_eqs, 1), dtype=np.float32))

        else:
            y_eqs = np.stack((xx, yy), axis=-1)  # shape = (Num_grid, Num_grid, 2)

            u_r_list.append(u.reshape(-1))
            y_r_list.append(y_eqs)
            s_r_list.append(np.ones((Num_grid, Num_grid, 1), dtype=np.float32))

    u = np.stack(u_r_list, axis=0)
    y = np.stack(y_r_list, axis=0)
    s = np.stack(s_r_list, axis=0)

    return torch.tensor(u), torch.tensor(y), torch.tensor(s)


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

    train_file = os.path.join('data', 'airfoil.npy')
    # valid_file = os.path.join('data', 'airfoil.npy')

    in_dim = 1
    out_dim = 1
    ntrain = 1000
    nvalid = 100
    batch_size = 32

    epochs = 1000
    learning_rate = 0.001
    scheduler_step = 300
    scheduler_gamma = 0.1

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    ################################################################
    # load data
    ################################################################

    raw_data = np.load(train_file, allow_pickle=True)
    raw_data = np.array(raw_data, dtype=np.float32)

    train_f, train_grid, train_u = gen_data(raw_data, 'train', 0, ntrain)
    valid_f, valid_grid, valid_u = gen_data(raw_data, 'valid', ntrain, nvalid)

    # f_normalizer = DataNormer(raw_data.reshape(raw_data.shape[0], -1), method='mean-std', axis=(0,))
    # train_f = f_normalizer.norm(train_f)
    # valid_f = f_normalizer.norm(valid_f)

    operator_dim = train_f.shape[1]

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
                    planes_branch=[64] * 4, planes_trunk=[64] * 4).to(Device)
    # 损失函数
    Loss_func = nn.MSELoss()
    # L1loss = nn.SmoothL1Loss()
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
        loss_tol, loss_eqs, loss_bcs = train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler, [253, ])
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


            fig = plt.figure(2, figsize=(18, 6))
            plt.clf()
            for fig_id in range(4):
                ref = valid_source[fig_id].reshape((-1, 2))

                # S_pred = griddata(y_star, valid_pred[fig_id].flatten(), (y1, y2), method='cubic')
                mask = np.abs(valid_pred[fig_id, ..., 0]) < 5e-3

                plt.subplot(2, 4, fig_id + 1)
                plt.plot(ref[:, 0], ref[:, 1], '--', label='Exact', color='blue')
                plt.plot(valid_coord[fig_id, ..., 0][mask], valid_coord[fig_id, ..., 1][mask], '.',
                         markersize=4, label='Predicted', color='red')
                plt.xlabel('$x$')
                plt.ylabel('$y$')
                plt.tight_layout()
                plt.title('Input sample')

                plt.subplot(2, 4, fig_id + 5)
                plt.pcolor(valid_coord[fig_id, ..., 0], valid_coord[fig_id, ..., 1], valid_pred[fig_id,  ..., 0],
                           cmap='RdYlBu_r')
                # plt.colorbar()
                plt.clim(-2., 2.)
                plt.xlabel('$x$')
                plt.ylabel('$y$')
                plt.tight_layout()
                plt.title('Predicted SDF')
            plt.savefig(os.path.join(work_path, 'valid_pred.jpg'))