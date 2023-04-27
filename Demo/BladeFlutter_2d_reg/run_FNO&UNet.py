#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/2/21 12:14
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : run_FNO&UNet.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from fno.FNOs import FNO2d
from cnn.ConvNets import UNet2d
from Utilizes.visual_data import MatplotlibVision, TextLogger
from Utilizes.process_data import DataNormer
from Utilizes.loss_metrics import FieldsLpLoss

import matplotlib.pyplot as plt
import time
import os
import h5py
import sys


def feature_transform(x):
    """
    Args:
        x: input coordinates
    Returns:
        res: input transform
    """
    shape = x.shape
    batchsize, size_x, size_y= shape[0], shape[1], shape[2]
    gridx = torch.linspace(0, 1, size_x, dtype=torch.float32)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.linspace(0, 1, size_y, dtype=torch.float32)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(x.device)


def train(dataloader, nets, device, lossfunc, optimizer, scheduler):
    """
    Args:
        data_loader: output fields at last time step
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """
    train_loss = 0
    for batch, (xx, yy, dd) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)
        dd = dd.to(device)
        gd = feature_transform(xx)

        fp, fs = nets[0](dd, gd), nets[1](dd, gd)
        pred = nets[-1](torch.concat((fp, fs), dim=-1), gd)
        loss = lossfunc(pred, yy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1) / batch_size


def valid(dataloader, nets, device, lossfunc):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """
    valid_loss = 0
    with torch.no_grad():
        for batch, (xx, yy, dd) in enumerate(dataloader):
            xx = xx.to(device)
            yy = yy.to(device)
            dd = dd.to(device)
            gd = feature_transform(xx)

            fp, fs = nets[0](dd, gd), nets[1](dd, gd)
            pred = nets[-1](torch.concat((fp, fs), dim=-1), gd)
            loss = lossfunc(pred, yy)
            valid_loss += loss.item()

    return valid_loss / (batch + 1) / batch_size


def inference(dataloader, nets, device):
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """

    with torch.no_grad():
        xx, yy, dd = next(iter(dataloader))
        xx = xx.to(device)
        yy = yy.to(device)
        dd = dd.to(device)
        gd = feature_transform(xx)

        fp, fs = nets[0](dd, gd), nets[1](dd, gd)
        pred = nets[-1](torch.concat((fp, fs), dim=-1), gd)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), gd.cpu().numpy(), yy.cpu().numpy(), pred.cpu().numpy()


class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor, design_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.design_tensor = design_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index], self.design_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)    # size(0) 返回当前张量维数的第一维


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'FNO'
    work_path = os.path.join('work', name + 'Reg')
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    sys.stdout = TextLogger(os.path.join(work_path, 'train.log'), sys.stdout)
    print(work_path)

    in_dim = 6 # 4(design) + 3(coord)  在FNO模型中会再次+ 2(grid as image)
    out_dim = 31 # p, t, entropy, modeshape xyzFNO
    ntrain = 200
    nvalid = 50

    modes = (64, 16)
    width = 32
    depth = 4
    steps = 1
    padding = 8
    dropout = 0.0

    batch_size = 32
    epochs = 5000
    learning_rate = 0.001
    scheduler_step = int(epochs*0.7)
    scheduler_gamma = 0.1

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    r1 = 3 #
    r2 = 1
    s1 = int(((794 - 1) / r1) + 1)
    s2 = int(((40 - 1) / r2) + 1)

    ################################################################
    # load data
    ################################################################

    data_file = os.path.join('..\BladeFlutter_2d\data', "flutter_bld_res1_1-300.mat")

    datamat = h5py.File(data_file)
    bld_fields= []
    index = []
    tmp = np.transpose(datamat['icm_damp'][:], (1, 0))[:, ::6]
    for ind, element in enumerate(datamat['NEW_bld_fields']):
        if np.size(datamat[element[0]][:]) > 10 and ~(np.any(np.isnan(tmp[ind, :]))):
            bld_fields.append(datamat[element[0]][:])
            index.append(ind)
    # AFPs = np.transpose(np.array(datamat['AFPs']), (1, 0))[index]
    damps = torch.from_numpy(tmp.astype(np.float32)[index])
    all_fields = torch.cat([F.interpolate(torch.tensor(ff[None, ...], dtype=torch.float32), [164, 36]) for ff in bld_fields], dim=0)
    fields = torch.permute(all_fields[:, 4:, ...], (0, 2, 3, 1))
    coords = torch.permute(all_fields[:, 1:4, ...], (0, 2, 3, 1))
    # bld_elems = [datamat[element[0]][:] for element in datamat['NEW_bld_nodes']] # 在图卷积中使用

    design1 = torch.tensor(np.transpose(datamat['boundaries'][:], (1, 0)).squeeze()[:, 3:5], dtype=torch.float32)
    design2 = torch.tensor(np.transpose(datamat['geometries'][:], (1, 0)).squeeze()[:, 2:4], dtype=torch.float32)
    design = torch.cat([design1, design2], 1)
    design = torch.tile(design[index, None, None, :], (1, coords[0].shape[0], coords[0].shape[1], 1))
    descrd = torch.concat([coords, design], dim=-1)
    input = fields
    output = torch.tile(damps[:, None, None, :], (1, fields[0].shape[0], fields[0].shape[1], 1))
    print(input.shape, output.shape)


    del datamat, design, design1, design2, all_fields, bld_fields

    # train_x = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
    # train_y = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    # valid_x = input[ntrain:ntrain + nvalid, ::r1, ::r2][:, :s1, :s2]
    # valid_y = output[ntrain:ntrain + nvalid, ::r1, ::r2][:, :s1, :s2]
    train_x = input[:ntrain, ...]
    train_y = output[:ntrain, ...]
    valid_x = input[ntrain:, ...]
    valid_y = output[ntrain:, ...]
    train_d = descrd[:ntrain, ...]
    valid_d = descrd[ntrain:, ...]

    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)

    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)

    d_normalizer = DataNormer(train_d.numpy(), method='mean-std')
    train_d = d_normalizer.norm(train_d)
    valid_d = d_normalizer.norm(valid_d)

    train_loader = torch.utils.data.DataLoader(TensorDataset(train_x, train_y, train_d, ),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(TensorDataset(valid_x, valid_y, valid_d, ),
                                               batch_size=batch_size, shuffle=False, drop_last=False)
    in_sizes = [train_x.shape[1], train_x.shape[2], train_x.shape[3]]
    ################################################################
    # Neural Networks
    ################################################################

    # 建立网络
    if name == 'FNO':
        PTE_Net = FNO2d(in_dim=7, out_dim=3, modes=(64, 16), width=32, depth=4, steps=1,
                          padding=8, activation='gelu', dropout=0).to(Device)
        ck = torch.load(r'..\BladeFlutter_2d\work\FNOForPTEMode\latest_model.pth')
        PTE_Net.load_state_dict(ck['net_model'])
        Sld_Net = FNO2d(in_dim=7, out_dim=3, modes=(64, 16), width=32, depth=4, steps=1,
                          padding=8, activation='gelu', dropout=0).to(Device)
        ck = torch.load(r'..\BladeFlutter_2d\work\FNOForSolid\latest_model.pth')
        Sld_Net.load_state_dict(ck['net_model'])
        # Net_model = FNO2d(in_dim=in_dim, out_dim=out_dim, modes=modes, width=width, depth=depth, steps=steps,
        #                   padding=padding, activation='gelu', dropout=dropout).to(Device)
        Net_model = UNet2d(in_sizes=(164, 36, 6), out_sizes=(164, 36, 31), width=width,
                           depth=depth, steps=steps, activation='gelu', dropout=dropout).to(Device)
    elif name == 'UNet':
        PTE_Net = UNet2d(in_sizes=(164, 36, 7), out_sizes=(164, 36, 3), width=32,
                           depth=4, steps=1, activation='gelu', dropout=0.0).to(Device)
        ck = torch.load(r'..\BladeFlutter_2d\work\UNetForPTE\latest_model.pth')
        PTE_Net.load_state_dict(ck['net_model'])
        Sld_Net = UNet2d(in_sizes=(164, 36, 7), out_sizes=(164, 36, 3),width=32,
                           depth=4, steps=1, activation='gelu', dropout=0.0).to(Device)
        ck = torch.load(r'..\BladeFlutter_2d\work\UNetForSolid\latest_model.pth')
        Sld_Net.load_state_dict(ck['net_model'])
        Net_model = UNet2d(in_sizes=(164, 36, 6), out_sizes=(164, 36, 31), width=width,
                           depth=depth, steps=steps, activation='gelu', dropout=dropout).to(Device)

    input1 = torch.randn(batch_size, train_x.shape[1], train_x.shape[2], train_x.shape[3]).to(Device)
    input2 = torch.randn(batch_size, train_x.shape[1], train_x.shape[2], 2).to(Device)

    summary(Net_model, input_data=[input1, input2], device=Device)

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
    start_epoch = 0


    # 读入模型
    if os.path.exists(os.path.join(work_path, 'latest_model.pth')):
        ck = torch.load(os.path.join(work_path, 'latest_model.pth'))
        Net_model.load_state_dict(ck['net_model'])
        log_loss = ck['log_loss']
        Optimizer.load_state_dict(ck['optimizer'])
        start_epoch = np.min([epochs, len(log_loss[0]) - 1])

    ################################################################
    # train process
    ################################################################
    nets = [PTE_Net, Sld_Net, Net_model]
    for epoch in range(start_epoch, epochs+1):

        Net_model.train()
        log_loss[0].append(train(train_loader, nets, Device, Loss_func, Optimizer, Scheduler))

        Net_model.eval()
        log_loss[1].append(valid(valid_loader, nets, Device, Loss_func))
        print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, Optimizer.param_groups[0]['lr'], log_loss[0][-1], log_loss[1][-1], time.time() - star_time))

        star_time = time.time()

        if epoch > 0 and epoch % 100 == 0:
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

            _, _, valid_true, valid_pred = inference(valid_loader, nets, Device)

            Error_func.p = 1
            ErrL1a = Error_func.abs(valid_pred, valid_true)
            ErrL1r = Error_func.rel(valid_pred, valid_true)
            Error_func.p = 2
            ErrL2a = Error_func.abs(valid_pred, valid_true)
            ErrL2r = Error_func.rel(valid_pred, valid_true)

            fig, axs = plt.subplots(2, 1, figsize=(10, 10), layout='constrained', num=3)
            Visual.plot_box(fig, axs[0], ErrL1r, legends=['damping',])
            Visual.plot_box(fig, axs[1], ErrL2r, legends=['damping',])
            fig.savefig(os.path.join(work_path, 'valid_box.jpg'))
            plt.close(fig)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict(),
                        'valid_true': valid_true, 'valid_pred': valid_pred,
                        'ErrL1a': ErrL1a, 'ErrL1r': ErrL1r, 'ErrL2a': ErrL2a, 'ErrL2r': ErrL2r,
                        },
                       os.path.join(work_path, 'latest_model.pth'))

            _, _, train_true, train_pred = inference(train_loader, nets, Device)
            # train_coord = x_normalizer.back(train_coord[-10:])
            # valid_coord = x_normalizer.back(valid_coord[-10:])
            train_true, valid_true = np.mean(y_normalizer.back(train_true[-10:]), axis=(1, 2)), np.mean(y_normalizer.back(valid_true[-10:]), axis=(1, 2))
            train_pred, valid_pred = np.mean(y_normalizer.back(train_pred[-10:]), axis=(1, 2)), np.mean(y_normalizer.back(valid_pred[-10:]), axis=(1, 2))
            for fig_id in range(1, 11):
                fig, axs = plt.subplots(1, 1, figsize=(10, 8), layout='constrained', num=20+fig_id)
                axs.plot(train_true[-fig_id], '-', label='true')
                axs.plot(train_pred[-fig_id], '*', label='pred')
                axs.set_title('icm damping')
                fig.savefig(os.path.join(work_path, 'train_damping_' + str(fig_id) + '.jpg'))
                plt.close(fig)

                fig, axs = plt.subplots(1, 1, figsize=(10, 8), layout='constrained', num=50+fig_id)
                axs.plot(valid_true[-fig_id], '-', label='true')
                axs.plot(valid_pred[-fig_id], '*', label='pred')
                axs.set_title('icm damping')
                fig.savefig(os.path.join(work_path, 'valid_damping_' + str(fig_id) + '.jpg'))
                plt.close(fig)





