#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/7/9 16:56
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : utilize.py
"""

import torch
import numpy as np
import torch.nn as nn

def data_preprocess(reader):
    """
        data preprocess
        :param file_loader: Mat loader
    """
    design = np.transpose(reader.read_field('Aphis_'), (1, 0))
    fields = np.transpose(reader.read_field('fields'), (5, 3, 0, 1, 4, 2))[:, :, :, ::2]  # (N, T, H, W, I, F)
    coords_x = fields[..., (0, 1, 2)]
    fields = fields[..., (3, 4)]
    target = np.transpose(reader.read_field('MFs'), (2, 0, 1))

    coords_t = torch.linspace(0, 2*np.pi, fields.shape[1], dtype=torch.float32)[None, :, None, None, None]\
                .repeat((fields.shape[0],) + (1,) + tuple(fields.shape[2:-1])).unsqueeze(-1)
    coords = torch.cat((coords_x, coords_t), dim=-1)

    np.random.seed(2023)
    index = np.random.permutation(np.arange(fields.shape[0]))

    return design[index], fields[index], coords[index], target[index]

from torch.utils.data import Dataset
class custom_dataset(Dataset):
    def __init__(self, design, coords, fields, target):
        self.design = design
        self.coords = coords
        self.fields = fields
        self.target = target
        self.sample_size = self.design.shape[0]
        self.time_size = self.coords.shape[1]

    def __getitem__(self, idx):  # 根据 idx 取出其中一个

        idt, ids = divmod(idx, self.sample_size)
        d = self.design[ids]
        c = self.coords[ids, idt]
        f = self.fields[ids, idt]
        t = self.target[ids, idt]
        return d, c, f, t

    def __len__(self):  # 总数据的多少
        return self.sample_size * self.time_size


class predictor(nn.Module):

    def __init__(self, branch, trunc, infer, field_dim, infer_dim):

        super(predictor, self).__init__()

        self.branch_net = branch
        self.trunc_net = trunc
        self.infer_net = infer
        self.field_net = nn.Linear(trunc.out_dim, field_dim)


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
        feature = B * T
        F = self.field_net(feature)
        Y = self.infer_net(feature)
        return F, Y


def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    """
    Args:
        data_loader: output fields at last time step
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """
    train_loss = np.array([0, 0], dtype=np.float32)
    total_size = 0
    history_loss = []
    for batch, (dd, xx, yy, tt) in enumerate(dataloader):
        input_sizes = list(xx.shape)
        dd = dd.to(device)
        xx = xx.to(device)
        yy = yy.to(device)
        tt = tt.to(device)

        fields, target = netmodel(dd, xx)
        fields_loss = lossfunc(fields, yy)
        target_loss = lossfunc(target, tt)
        total_loss = fields_loss + target_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_loss[0] += fields_loss.item() * input_sizes[0]
        train_loss[1] += target_loss.item() * input_sizes[0]
        total_size += input_sizes[0]
        history_loss.append([fields_loss.item(), target_loss.item()])

    scheduler.step()
    return train_loss / total_size, history_loss


def valid(dataloader, netmodel, device, lossfunc):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """
    valid_loss = np.array([0, 0], dtype=np.float32)
    total_size = 0
    history_loss = []
    with torch.no_grad():
        for batch, (dd, xx, yy, tt) in enumerate(dataloader):
            input_sizes = list(xx.shape)
            dd = dd.to(device)
            xx = xx.to(device)
            yy = yy.to(device)
            tt = tt.to(device)

            fields, target = netmodel(dd, xx)
            fields_loss = lossfunc(fields, yy)
            target_loss = lossfunc(target, tt)
            valid_loss[0] += fields_loss.item() * input_sizes[0]
            valid_loss[1] += target_loss.item() * input_sizes[0]
            total_size += input_sizes[0]
            history_loss.append([fields_loss.item(), target_loss.item()])

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
        dd, xx, yy, tt = next(iter(dataloader))
        dd = dd.to(device)
        xx = xx.to(device)
        fields, target = netmodel(dd, xx)

    # equation = model.equation(u_var, y_var, out_pred)
    return dd.cpu().numpy(), xx.cpu().numpy(), yy.numpy(), tt.numpy(), fields.cpu().numpy(), target.cpu().numpy()


def cal_damps(mfs):
    device = mfs.device
    vibra_f = 207.24421
    dis = 59.713451
    pi = torch.pi
    n_blade = 30
    n_time = 64
    # passage = 5
    IBPAs = (torch.linspace(0, 360, n_blade+1, device=device) - 180) / 180 * pi

    mfs = mfs[:, -n_time - 1:]
    E = 2 * pi * (1e-3 / dis) ** 2 * (2 * pi * vibra_f) ** 2
    t = torch.linspace(0, 1, n_time+1, device=device) / vibra_f
    # y = np.fft.fft(mfs, n_time, axis=1)
    y = torch.fft.fft(mfs, n_time, dim=1)
    Ayy = torch.abs(y) / (n_time / 2)
    Ayy[:, 0] = Ayy[:, 0] / 2
    Fyy = torch.arange(0, n_time, device=device) * vibra_f
    Pyy = torch.angle(y)

    t = t[None, :, None, None]
    A0, A1, F1, P1 = Ayy[:, 0:1, :, None], Ayy[:, 1:2, :, None], Fyy[None, 1:2, None, None], Pyy[:, 1:2, :, None]
    p = torch.tensor([-2, -1, 0, 1, 2])[None, None, :, None].to(device)

    mf = A0 + A1 * torch.cos(2 * pi * F1 * t + P1 - p * IBPAs)
    wt = - (1e-3 / dis) * 2 * pi * F1 * torch.cos(2 * pi * F1 * t) * mf
    wk = torch.trapz(wt, t, dim=1).sum(dim=-2)
    damps = -wk / E

    AFP = torch.cat([A0[:, 0, :], A1[:, 0, :], P1[:, 0, :]], dim=2)
    return damps.cpu().numpy()

if __name__ == "__main__":
    from process_data import MatLoader

    data_path = 'data/y_bend-10_RN1_A5_APhis_trans_field.mat'
    reader = MatLoader(data_path)
    MFs = reader.read_field('MFs').permute((2, 0, 1))
    damps = reader.read_field('damps').permute((1, 0))[:, ::6]

    damps_pred = cal_damps(MFs)

    print(torch.sum(torch.abs(damps_pred - damps)))