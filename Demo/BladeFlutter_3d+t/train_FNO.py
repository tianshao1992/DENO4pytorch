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
from cnn.ConvNets import DownSampleNet2d
from Utilizes.process_data import DataNormer
from Utilizes.visual_data import MatplotlibVision, TextLogger

import matplotlib.pyplot as plt
import time
import os
from torchinfo import summary
from shutil import copyfile
from operator import add
from functools import reduce

from utilize import predictor, train, valid, inference, data_preprocess, custom_dataset


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'FNO+MLP+CNN-1'
    work_path = os.path.join('work', name)
    train_path = os.path.join(work_path, 'train')
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)
        os.makedirs(train_path)

    # 将控制台的结果输出到log文件
    Logger = TextLogger(os.path.join(train_path, 'train.log'))

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')
    Logger.info("Model Name: {:s}, Computing Device: {:s}".format(name, str(Device)))

    ntrain = 20
    nvalid = 5

    batch_size = 32
    epochs = 301
    learning_rate = 0.001
    scheduler_step = 201
    scheduler_gamma = 0.1

    Logger.info('Total epochs: {:d}, learning_rate: {:e}, scheduler_step: {:d}, scheduler_gamma: {:.3e}'
                .format(epochs, learning_rate, scheduler_step, scheduler_gamma))

    ################################################################
    # load data
    ################################################################
    data_path = 'data/y_bend-10_RN1_A5_APhis_trans_field.mat'
    reader = MatLoader(data_path)
    design, fields, coords, target = data_preprocess(reader)
    coords = coords.reshape(tuple(coords.shape[0:-2]) + (-1,))
    fields = fields.reshape(tuple(fields.shape[0:-2]) + (-1,))

    design_normer = DataNormer(design)
    coords_normer = DataNormer(coords)
    fields_normer = DataNormer(fields, method='mean-std')
    target_normer = DataNormer(target, method='mean-std')
    design = design_normer.norm(design)
    coords = coords_normer.norm(coords)
    fields = fields_normer.norm(fields)
    target = target_normer.norm(target)

    train_dataset = custom_dataset(design[:ntrain], coords[:ntrain], fields[:ntrain], target[:ntrain])
    valid_dataset = custom_dataset(design[-nvalid:], coords[-nvalid:], fields[-nvalid:], target[-nvalid:])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size, shuffle=False, drop_last=False)

    from utilize import cal_damps
    plt.figure(10)
    plt.clf()
    plt.scatter(design_normer.back(design[:ntrain]),
                cal_damps(target_normer.back(target[:ntrain])).min(axis=-1), label='train')
    plt.scatter(design_normer.back(design[-nvalid:]),
                cal_damps(target_normer.back(target[-nvalid:])).min(axis=-1), label='valid')
    plt.legend()
    plt.savefig(os.path.join(work_path, 'train_test_split.jpg'))

    del reader

    ################################################################
    # Neural Networks
    ################################################################

    FNO_model = FNO2d(in_dim=20, out_dim=64, modes=(16, 16), width=64, depth=4,
                      padding=5, activation='gelu').to(Device)
    MLP_model = FcnSingle(planes=(1, 64, 64, 64), last_activation=True).to(Device)
    CNN_model = DownSampleNet2d(in_sizes=tuple(fields.shape[1:-1]) + (64,), out_sizes=target.shape[-1],
                                width=64, depth=4, dropout=0.05).to(Device)

    Net_model = predictor(trunc=FNO_model, branch=MLP_model, infer=CNN_model,
                          field_dim=10, infer_dim=target.shape[-1]).to(Device)


    (dd, xx, yy, tt) = next(iter(train_loader))
    input_sizes = list(xx.shape)
    dd = dd.to(Device)
    xx = xx.to(Device)
    model_statistics = summary(Net_model, input_data=[dd, xx], device=Device, verbose=0)
    Logger.write(str(model_statistics))

    # 损失函数
    Loss_func = nn.MSELoss()
    # L1loss = nn.SmoothL1Loss()
    # Loss_metirc = FieldsLpLoss(d=2, p=2, reduction=True, size_average=False)
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=0.0)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    field_name = reduce(add, [['P_' + str(i), 'M_' + str(i)] for i in range(1, 6)])
    Visual = MatplotlibVision(train_path, input_name=('x', 'y'), field_name=field_name)

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
                    'train_fields_loss: {:.3e}, valid_fields_loss: {:.3e}, '
                    'train_target_loss: {:.3e}, valid_target_loss: {:.3e}, '
                    'cost: {:.2f}'.
                    format(epoch, Optimizer.param_groups[0]['lr'],
                           log_loss[0][-1][0], log_loss[1][-1][0], log_loss[0][-1][1], log_loss[1][-1][1],
                           time.time() - star_time))

        star_time = time.time()

        if epoch > 0 and epoch % 10 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(14, 10), num=1)
            Visual.plot_value(fig, axs, np.arange(len(log_loss[0])),
                              np.array(log_loss[0])[:, 0], std=np.array(log_per[0])[..., 0].std(axis=1),
                              label='训练集物理场预测损失', xylabels=('迭代步', '损失函数'))
            Visual.plot_value(fig, axs, np.arange(len(log_loss[1])),
                              np.array(log_loss[1])[:, 0], std=np.array(log_per[1])[..., 0].std(axis=1), std_factor=5.0,
                              label='验证集物理场预测损失', xylabels=('迭代步', '损失函数'))
            Visual.plot_value(fig, axs, np.arange(len(log_loss[0])),
                              np.array(log_loss[0])[:, 1], std=np.array(log_per[0])[..., 1].std(axis=1),
                              label='训练集模态力预测损失',  xylabels=('迭代步', '损失函数'))
            Visual.plot_value(fig, axs, np.arange(len(log_loss[1])),
                              np.array(log_loss[1])[:, 1], std=np.array(log_per[1])[..., 1].std(axis=1),
                              label='验证集模态力预测损失', xylabels=('迭代步', '损失函数'))
            axs.semilogy(np.arange(len(log_loss[0])), np.array(log_loss[1])[:, 0])

            fig.suptitle('训练过程损失函数收敛过程', font=Visual.font)
            fig.savefig(os.path.join(train_path, 'log_loss.svg'))
            plt.close(fig)


        if epoch > 0 and epoch % 10 == 0:
            save_file = os.path.join(train_path, 'epoch_' + str(epoch) + '.pth')
            torch.save({'log_loss': log_loss, 'log_per': log_per,
                        'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()}, save_file)
            copyfile(save_file, os.path.join(train_path, 'latest_model.pth'))

            train_design, train_coords, train_fields_t, train_target_t, train_fields_p, train_target_p \
                = inference(train_loader, Net_model, Device)
            valid_design, valid_coords, valid_fields_t, valid_target_t, valid_fields_p, valid_target_p \
                = inference(valid_loader, Net_model, Device)

            train_design = design_normer.back(train_design)
            valid_design = design_normer.back(valid_design)
            train_coords = coords_normer.back(train_coords)
            valid_coords = coords_normer.back(valid_coords)
            train_fields_t, train_fields_p = fields_normer.back(train_fields_t), fields_normer.back(train_fields_p)
            valid_fields_t, valid_fields_p = fields_normer.back(valid_fields_t), fields_normer.back(valid_fields_p)
            train_target_t, train_target_p = target_normer.back(train_target_t), target_normer.back(train_target_p)
            valid_target_t, valid_target_p = target_normer.back(valid_target_t), target_normer.back(valid_target_p)

            for tar_id in range(valid_target_p.shape[-1]):
                fig, axs = plt.subplots(2, 2, figsize=(16, 10), layout='constrained', num=5)
                Visual.plot_regression(fig, axs[0, 0], train_target_t[..., tar_id], train_target_p[..., tar_id],
                                       error_ratio=0.005, title='训练集', xylabels=('真实值', '预测值'))
                Visual.plot_regression(fig, axs[1, 0], valid_target_t[..., tar_id], valid_target_p[..., tar_id],
                                       error_ratio=0.005, title='验证集', xylabels=('真实值', '预测值'))
                err = (train_target_p[..., tar_id] - train_target_t[..., tar_id]) / train_target_t[..., tar_id]
                Visual.plot_error(fig, axs[0, 1], err, error_ratio=0.005, xylabels=('预测偏差/ %', '分布密度'))
                err = (valid_target_p[..., tar_id] - valid_target_t[..., tar_id]) / valid_target_t[..., tar_id]
                Visual.plot_error(fig, axs[1, 1], err, error_ratio=0.005, xylabels=('预测偏差/ %', '分布密度'))
                fig.savefig(os.path.join(train_path, 'target_prediction_' + str(tar_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(5):
                fig, axs = plt.subplots(2, 3, figsize=(20, 10), layout='constrained', num=2)
                Visual.plot_fields_ms(fig, axs, train_fields_t[fig_id], train_fields_p[fig_id], show_channel=(4, 5))
                fig.suptitle('design: {:.2f} time: {:.2f}'.
                                      format(float(train_design[fig_id]), float(train_coords[fig_id, 0, 0, 0])))
                fig.savefig(os.path.join(train_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(5):
                fig, axs = plt.subplots(2, 3, figsize=(20, 10), layout='constrained', num=3)
                Visual.plot_fields_ms(fig, axs, valid_fields_t[fig_id], valid_fields_p[fig_id], show_channel=(4, 5))
                fig.suptitle('design: {:.2f} time: {:.2f}'.format
                             (float(valid_design[fig_id]), float(valid_coords[fig_id, 0, 0, 0])))
                fig.savefig(os.path.join(train_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)



