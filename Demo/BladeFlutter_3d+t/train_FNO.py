#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/7/8 17:07
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : train_FNO.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Utilizes.process_data import MatLoader
from fno.FNOs import FNO2d
from basic.basic_layers import FcnSingle
from Utilizes.process_data import DataNormer
from Utilizes.visual_data import MatplotlibVision, TextLogger
from Utilizes.loss_metrics import FieldsLpLoss

import matplotlib.pyplot as plt
import time
import os
from torchinfo import summary
from shutil import copyfile
from tqdm import tqdm
from operator import add
from functools import reduce


def data_preprocess(reader):
    """
        data preprocess
        :param file_loader: Mat loader
    """
    design = np.transpose(reader.read_field('Aphis_'), (1, 0))
    fields = np.transpose(reader.read_field('fields'), (5, 3, 0, 1, 4, 2))  # (N, T, H, W, I, F)
    coords_x = fields[..., (0, 1, 2)]
    fields = fields[..., (3, 4)]

    coords_t = torch.linspace(0, 2*np.pi, fields.shape[1], dtype=torch.float32)[None, :, None, None, None]\
                .repeat((fields.shape[0],) + (1,) + tuple(fields.shape[2:-1])).unsqueeze(-1)
    coords = torch.cat((coords_x, coords_t), dim=-1)

    np.random.seed(2022)
    index = np.random.permutation(np.arange(fields.shape[0]))

    return design[index], fields[index, :, :, ::2], coords[index, :, :, ::2]

from torch.utils.data import Dataset
class custom_dataset(Dataset):
    def __init__(self, design, coords, fields):
        self.design = design
        self.coords = coords
        self.fields = fields
        self.sample_size = self.design.shape[0]
        self.time_size = self.coords.shape[1]

    def __getitem__(self, idx):  # 根据 idx 取出其中一个

        idt, ids = divmod(idx, self.sample_size)
        d = self.design[ids]
        c = self.coords[ids, idt]
        f = self.fields[ids, idt]
        return d, c, f

    def __len__(self):  # 总数据的多少
        return self.sample_size * self.time_size


class predictor(nn.Module):

    def __init__(self, FNO, MLP, output_dim):

        super(predictor, self).__init__()

        self.branch_net = MLP
        self.trunc_net = FNO
        self.infer_net = nn.Linear(FNO.out_dim, output_dim)

    def forward(self, design, coords):
        """
        forward compute
        :param design: tensor list[(batch_size, ..., operator_dims[0]), (batch_size, ..., operator_dims[1]), ...]
        :param coords: (batch_size, ..., input_dim)
        """

        T = self.trunc_net(coords)
        B = self.branch_net(design)
        T_size = T.shape[1:-1]
        for i in range(len(T_size)):
            B = B.unsqueeze(1)
        B = torch.tile(B, [1, ] + list(T_size) + [1, ])
        F = self.infer_net(B * T)
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
    total_size = 0
    history_loss = []
    for batch, (dd, xx, yy) in enumerate(dataloader):
        input_sizes = list(xx.shape)
        dd = dd.to(device)
        xx = xx.to(device)
        yy = yy.to(device)

        pred = netmodel(dd, xx)
        loss = lossfunc(pred, yy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * input_sizes[0]
        total_size += input_sizes[0]
        history_loss.append(loss.item())

    scheduler.step()
    return train_loss / total_size, history_loss


def valid(dataloader, netmodel, device, lossfunc):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """
    valid_loss = 0
    total_size = 0
    history_loss = []
    with torch.no_grad():
        for batch, (dd, xx, yy) in enumerate(dataloader):
            input_sizes = list(xx.shape)
            dd = dd.to(device)
            xx = xx.to(device)
            yy = yy.to(device)

            pred = netmodel(dd, xx)
            loss = lossfunc(pred, yy)
            valid_loss += loss.item() * input_sizes[0]
            total_size += input_sizes[0]
            history_loss.append(loss.item())

    return valid_loss / total_size, history_loss


def inference(dataloader, netmodel, device):
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """

    with torch.no_grad():
        dd, xx, yy = next(iter(dataloader))
        dd = dd.to(device)
        xx = xx.to(device)
        pred = netmodel(dd, xx)

    # equation = model.equation(u_var, y_var, out_pred)
    return dd.cpu().numpy(), xx.cpu().numpy(), yy.numpy(), pred.cpu().numpy()





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
    Logger = TextLogger(os.path.join(work_path, 'train.log'))

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')
    Logger.info("Model Name: {:s}, Computing Device: {:s}".format(name, str(Device)))

    ntrain = 20
    nvalid = 5

    batch_size = 32
    epochs = 151
    learning_rate = 0.001
    scheduler_step = 121
    scheduler_gamma = 0.1

    Logger.info('Total epochs: {:d}, learning_rate: {:e}, scheduler_step: {:d}, scheduler_gamma: {:e}'
                .format(epochs, learning_rate, scheduler_step, scheduler_gamma))

    ################################################################
    # load data
    ################################################################
    data_path = 'data/y_bend-10_RN1_A5_APhis_trans_field.mat'
    reader = MatLoader(data_path)
    design, fields, coords = data_preprocess(reader)
    coords = coords.reshape(tuple(coords.shape[0:-2]) + (-1,))
    fields = fields.reshape(tuple(fields.shape[0:-2]) + (-1,))

    design_normer = DataNormer(design)
    coords_normer = DataNormer(coords)
    fields_normer = DataNormer(fields, method='mean-std')
    design = design_normer.norm(design)
    coords = coords_normer.norm(coords)
    fields = fields_normer.norm(fields)

    train_dataset = custom_dataset(design[:ntrain], coords[:ntrain], fields[:ntrain])
    valid_dataset = custom_dataset(design[-nvalid:], coords[-nvalid:], fields[-nvalid:])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size, shuffle=False, drop_last=False)

    del reader

    FNO_model = FNO2d(in_dim=20, out_dim=64, modes=(16, 16), width=64, depth=4,
                      padding=5, activation='gelu').to(Device)
    MLP_model = FcnSingle(planes=(1, 64, 64, 64), last_activation=True).to(Device)
    Net_model = predictor(FNO_model, MLP_model, output_dim=10)


    (dd, xx, yy) = next(iter(train_loader))
    input_sizes = list(xx.shape)
    dd = dd.to(Device)
    xx = xx.to(Device)
    yy = yy.to(Device)
    model_statistics = summary(Net_model, input_data=[dd, xx], device=Device, verbose=0)
    Logger.write(str(model_statistics))

    # 损失函数
    Loss_func = nn.MSELoss()
    # L1loss = nn.SmoothL1Loss()
    Loss_metirc = FieldsLpLoss(d=2, p=2, reduction=True, size_average=False)
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-6)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    field_name = reduce(add, [['P_' + str(i), 'M_' + str(i)] for i in range(1, 6)])
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=field_name)

    star_time = time.time()
    log_loss = [[], []]
    log_per = [[], []]
    ################################################################
    # train process
    ################################################################

    for epoch in range(epochs):

        Net_model.train()
        train_loss, train_history = \
            train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler)
        log_loss[0].append(train_loss)
        log_per[0].append(train_history)

        Net_model.eval()
        valid_loss, valid_history = \
            valid(valid_loader, Net_model, Device, Loss_func)
        log_loss[1].append(valid_loss)
        log_per[1].append(valid_history)

        Logger.info('epoch: {:5d}, lr: {:.3e}, '
                    'train_step_loss: {:.3e}, valid_step_loss: {:.3e}, '
                    'cost: {:.2f}'.
                    format(epoch, Optimizer.param_groups[0]['lr'],
                           log_loss[0][-1], log_loss[1][-1],
                           time.time() - star_time))

        star_time = time.time()

        if epoch > 0 and epoch % 10 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(10, 8), num=1)
            # Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_avg')
            # Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_avg')
            Visual.plot_value(fig, axs, np.arange(len(log_loss[0])),
                              np.array(log_loss)[0, :],
                              std=np.array(log_per[0]).std(axis=1), label='train', xylabels=('epoch', 'loss'))
            Visual.plot_value(fig, axs, np.arange(len(log_loss[1])),
                              np.array(log_loss)[1, :], rangeIndex=10.0,
                              std=np.array(log_per[1]).std(axis=1), label='valid', xylabels=('epoch', 'loss'))
            axs.semilogy(np.arange(len(log_loss[0])), np.array(log_loss)[1, :])

            fig.suptitle('training loss')
            fig.savefig(os.path.join(work_path, 'log_loss.svg'))
            plt.close(fig)


        if epoch > 0 and epoch % 10 == 0:
            torch.save({'log_loss': log_loss, 'log_per': log_per,
                        'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'epoch_' + str(epoch) + '.pth'))
            copyfile(os.path.join(work_path, 'epoch_' + str(epoch) + '.pth'),
                                  os.path.join(work_path, 'latest_model.pth'))

            train_design, train_coord, train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_design, valid_coord, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

            train_design = design_normer.back(train_design)
            valid_design = design_normer.back(valid_design)
            train_coords = coords_normer.back(train_coord)
            valid_coords = coords_normer.back(valid_coord)
            train_true, valid_true = fields_normer.back(train_true), fields_normer.back(valid_true)
            train_pred, valid_pred = fields_normer.back(train_pred), fields_normer.back(valid_pred)

            for fig_id in range(5):
                fig, axs = plt.subplots(2, 3, figsize=(20, 10), layout='constrained', num=2)
                Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], show_channel=(4, 5))
                fig.suptitle('design: {:.2f} time: {:.2f}'.
                                      format(float(train_design[fig_id]), float(train_coords[fig_id, 0, 0, 0])))
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(5):
                fig, axs = plt.subplots(2, 3, figsize=(20, 10), layout='constrained', num=3)
                Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], show_channel=(4, 5))
                fig.suptitle('design: {:.2f} time: {:.2f}'.format
                             (float(valid_design[fig_id]), float(valid_coords[fig_id, 0, 0, 0])))
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)



