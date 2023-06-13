# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/6/6 11:19
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：run_FNO.py.py
@File ：run_FNO.py.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
from fno.FNOs import FNO3d
from cnn.ConvNets import UNet2d

from Utilizes.visual_data import MatplotlibVision

from Utilizes.visual_data import MatplotlibVision, TextLogger
from Utilizes.process_data import DataNormer

import matplotlib.pyplot as plt
import time
import sys


def feature_transform(x):
    """
    Args:
        x: input coordinates
    Returns:
        res: input transform
    """
    shape = x.shape
    batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
    gridx = torch.linspace(0, 1, size_x, dtype=torch.float32)
    gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    gridy = torch.linspace(0, 1, size_y, dtype=torch.float32)
    gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    gridz = torch.linspace(0, 1, size_z, dtype=torch.float32)
    gridz = gridz.reshape(1, 1, 1, size_y, 1).repeat([batchsize, size_x, size_y, 1, 1])
    return torch.cat((gridx, gridy, gridz), dim=-1).to(x.device)


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
    for batch, (xx, yy) in enumerate(dataloader):
        input_sizes = list(xx.shape)
        xx = xx.reshape(input_sizes[:-2] + [-1, ])
        xx = xx.to(device)
        yy = yy.to(device)
        gd = feature_transform(xx)

        pred = netmodel(xx, gd)
        loss = lossfunc(pred, yy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_size += input_sizes[0]

    scheduler.step()
    return train_loss / total_size


def valid(dataloader, netmodel, device, lossfunc):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """
    valid_loss = 0
    total_size = 0
    with torch.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            input_sizes = list(xx.shape)
            xx = xx.reshape(input_sizes[:-2] + [-1, ])
            xx = xx.to(device)
            yy = yy.to(device)
            gd = feature_transform(xx)

            pred = netmodel(xx, gd)
            loss = lossfunc(pred, yy)
            valid_loss += loss.item()
            total_size += input_sizes[0]

    return valid_loss / total_size


def inference(dataloader, netmodel, device):  # 这个是？？
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """

    with torch.no_grad():
        xx, yy = next(iter(dataloader))
        input_sizes = xx.shape
        xx = xx.reshape(input_sizes[:-2] + [-1, ])
        xx = xx.to(device)
        gd = feature_transform(xx)
        pred = netmodel(xx, gd)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), gd.cpu().numpy(), yy.numpy(), pred.cpu().numpy()


from torch.utils.data import Dataset


class custom_dataset(Dataset):
    def __init__(self, data, input_step):
        self.data = data
        self.input_step = input_step
        self.sample_size = data.shape[0]
        self.step_size = data.shape[1]

    def __getitem__(self, idx):  # 根据 idx 取出其中一个

        idt, ids = divmod(idx, self.sample_size)
        x = self.data[ids, idt:idt + self.input_step]
        y = self.data[ids, idt + self.input_step]
        return x.transpose((1, 2, 3, 4, 0)), y

    def __len__(self):  # 总数据的多少
        return self.sample_size * (self.step_size - self.input_step)


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
    #  torch.cuda.set_device(1)

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')
    Logger.info("Model Name: {:s}, Computing Device: {:s}".format(name, str(Device)))

    in_dim = 3
    out_dim = 3
    ntrain = 45
    nvalid = 5

    mode = 16
    modes = (mode, mode, mode)
    width = 64
    depth = 4
    steps = 5
    padding = 0
    dropout = 0.0

    batch_size = 16
    epochs = 501
    learning_rate = 0.001
    scheduler_step = 400
    scheduler_gamma = 0.1

    Logger.info('Total epochs: {:d}, learning_rate: {:e}, scheduler_step: {:d}, scheduler_gamma: {:e}'
                .format(epochs, learning_rate, scheduler_step, scheduler_gamma))

    r1 = 1
    r2 = 1
    r3 = 1
    s1 = int(((32 - 1) / r1) + 1)
    s2 = int(((32 - 1) / r2) + 1)
    s3 = int(((32 - 1) / r3) + 1)

    ################################################################
    # load data
    ################################################################

    all_data = np.load('data/HIT_vel_50g_600p_gap200_32.npy')  #
    train_data = all_data[:ntrain, :, ::r1, ::r2, ::r3][..., :s1, :s2, :s3, :]
    valid_data = all_data[ntrain:, :, ::r1, ::r2, ::r3][..., :s1, :s2, :s3, :]
    train_dataset = custom_dataset(train_data, input_step=steps)
    valid_dataset = custom_dataset(valid_data, input_step=steps)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size, shuffle=False, drop_last=False)

    ################################################################
    # Neural Networks
    ################################################################

    # 建立网络
    if 'FNO' in name:
        Net_model = FNO3d(in_dim=in_dim, out_dim=out_dim, modes=modes, width=width, depth=depth, steps=steps,
                          padding=padding, activation='gelu').to(Device)
    elif name == 'UNet':
        Net_model = UNet2d(in_sizes=(32, 32, 32, 5 * 3), out_sizes=(32, 32, 32, 3), width=width,
                           depth=depth, steps=steps, activation='gelu', dropout=dropout).to(Device)

    input1 = torch.randn(batch_size, 32, 32, 32, 5 * 3).to(Device)
    input2 = torch.randn(batch_size, 32, 32, 32, 3).to(Device)
    model_statistics = summary(Net_model, input_data=[input1, input2], device=Device, verbose=0)
    Logger.write(str(model_statistics))

    # 损失函数
    Loss_func = nn.MSELoss()
    # Loss_func = nn.SmoothL1Loss()
    # Loss_func = FieldsLpLoss(size_average=False)
    # L1loss = nn.SmoothL1Loss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-4)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('u', 'v', 'w'))

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
                    format(epoch, Optimizer.param_groups[0]['lr'], log_loss[0][-1], log_loss[1][-1],
                           time.time() - star_time))

        star_time = time.time()

        if epoch > 0 and epoch % 5 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(work_path, 'log_loss.svg'))
            plt.close(fig)
