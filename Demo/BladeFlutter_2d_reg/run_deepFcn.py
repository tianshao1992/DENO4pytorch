#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/2/4 0:15
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : run_deepFcn.py.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Utilizes.process_data import MatLoader
from basic.basic_layers import FcnMulti, FcnSingle
from pinn.differ_layers import gradients
from don.DeepONets import DeepONetMulti
from Utilizes.loss_metrics import FieldsLpLoss
from Utilizes.visual_data import MatplotlibVision, TextLogger
from Utilizes.process_data import DataNormer

import matplotlib.pyplot as plt
import time
import os
import sys
import h5py


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
    train_loss_eqs = 0
    train_loss_bcs = 0
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)
        optimizer.zero_grad()
        if 'ONet' in work_path:
            pred = netmodel([xx[..., 3:],], xx[..., :3])
        else:
            pred = netmodel(xx)
        loss = lossfunc(pred, yy)
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

            if 'ONet' in work_path:
                pred = netmodel(xx[..., 3:], xx[..., :3])
            else:
                pred = netmodel(xx)
            loss = lossfunc(pred, yy)
            valid_loss += loss.item()

    return valid_loss / (batch + 1)



def inference(xx, netmodel, device):
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """
    with torch.no_grad():
        xx = xx.to(device)
        if 'ONet' in work_path:
            pred = netmodel(xx[..., 3:], xx[..., :3])
        else:
            pred = netmodel(xx)

    # equation = model.equation(u_var, y_var, out_pred)
    return pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'DeepONet'
    work_path = os.path.join('work', name, '[32_5_6_]')
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

    in_dim = 7 # 4(design) + 3(coord)
    out_dim = 3 # p, t, entropy, modeshape xyzFNO
    ntrain = 200
    nvalid = 50

    batch_size = 1000

    epochs = 1000
    learning_rate = 0.001
    scheduler_step = 800
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
    output = fields
    print(input.shape, output.shape)
    train_x = torch.reshape(input[:ntrain, ...], (-1, input.shape[-1]))
    train_y = torch.reshape(output[:ntrain, ...], (-1, output.shape[-1]))
    valid_x = torch.reshape(input[ntrain:ntrain + nvalid, ...], (-1, input.shape[-1]))
    valid_y = torch.reshape(output[ntrain:ntrain + nvalid, ...], (-1, output.shape[-1]))

    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)

    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                               batch_size=164*36, shuffle=False, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                               batch_size=164*36, shuffle=False, drop_last=False)

    ################################################################
    #  Neural Networks
    ################################################################
    # 建立网络
    # Net_model = FcnSingle([in_dim] + [32, ]*7 + [out_dim]).to(Device)
    Net_model = DeepONetMulti(3, [4,], 3, [8, 16, 32, 16, 8], [8, 16, 32, 16, 8]).to(Device)
    # 损失函数
    Loss_func = nn.MSELoss()
    Error_func = FieldsLpLoss(size_average=False)
    # L1loss = nn.SmoothL1Loss()
    # 评价指标
    Field_metric = FieldsLpLoss(d=2, p=2, reduction=True, size_average=False)
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-4)
    # Optimizer = torch.optim.SGD(Net_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y',), field_name=('Pressure', 'Temperature', 'Static Entropy'))

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

        if epoch > 0 and epoch % 10 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(work_path, 'log_loss.svg'))
            plt.close(fig)

        ################################################################
        # Visualization
        ################################################################

        if epoch > 0 and epoch % 10 == 0:

            train_source, valid_source = input[:ntrain, ...][-10:], input[ntrain:, ...]
            train_true, valid_true = output[:ntrain, ...][-10:], output[ntrain:, ...]
            train_source_ = x_normalizer.norm(torch.reshape(train_source, (-1, input.shape[-1])))
            valid_source_ = x_normalizer.norm(torch.reshape(valid_source, (-1, input.shape[-1])))
            train_pred = inference(train_source_, Net_model, Device)
            valid_pred = inference(valid_source_, Net_model, Device)

            Error_func.p = 1
            ErrL1a = Error_func.abs(valid_pred, valid_true.numpy())
            ErrL1r = Error_func.rel(valid_pred, valid_true.numpy())
            Error_func.p = 2
            ErrL2a = Error_func.abs(valid_pred, valid_true.numpy())
            ErrL2r = Error_func.rel(valid_pred, valid_true.numpy())

            fig, axs = plt.subplots(1, 2, figsize=(20, 20), layout='constrained', num=3)
            Visual.plot_box(fig, axs[0], ErrL1r, legends=Visual.field_name)
            Visual.plot_box(fig, axs[1], ErrL2r, legends=Visual.field_name)
            fig.savefig(os.path.join(work_path, 'valid_box.jpg'))
            plt.close(fig)

            # err_rel = Field_metric.rel(valid_pred, valid_true)

            for fig_id in range(10):
                # 100 为验证集算例中的网格节点个数


                train_pred, valid_pred = y_normalizer.back(train_pred), y_normalizer.back(valid_pred)
                train_pred, valid_pred = train_pred.reshape(-1, 164, 36, 3), valid_pred.reshape(-1, 164, 36, 3)
                torch.save(
                    {'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict(),
                     'valid_coord': valid_source[..., :3], 'valid_true': valid_true, 'valid_pred': valid_pred,
                     'ErrL1a': ErrL1a, 'ErrL1r': ErrL1r, 'ErrL2a': ErrL2a, 'ErrL2r': ErrL2r,
                     },
                    os.path.join(work_path, 'latest_model.pth'))

                train_coord = x_normalizer.back(train_source)[..., :3]
                valid_coord = x_normalizer.back(valid_source)[..., :3]

                fig, axs = plt.subplots(3, 3, figsize=(18, 15), layout='constrained', num=3+fig_id)
                Visual.plot_fields_grid(fig, axs, valid_true.numpy()[-fig_id-1],
                                      valid_pred[-fig_id-1])
                # plt.title('Absolute error:  {:.3e}'.format(float(err_rel[fig_id])), fontdict=Visual.font_EN)
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)
                fig, axs = plt.subplots(3, 3, figsize=(18, 15), layout='constrained', num=20+fig_id)
                Visual.plot_fields_grid(fig, axs, train_true.numpy()[-fig_id-1],
                                      train_pred[-fig_id-1])
                # plt.title('Absolute error:  {:.3e}'.format(float(err_rel[fig_id])), fontdict=Visual.font_EN)
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)
                Visual.output_tecplot_struct(valid_true.numpy(), valid_pred.reshape(valid_true.shape),
                                             valid_source.numpy()[..., :3],
                                             ['Pressure', 'Temperature', 'Static Entropy'],
                                             os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.dat'))