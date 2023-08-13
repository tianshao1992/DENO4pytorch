#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/7/16 19:18
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : run_.py
# @Description    : 标准化的数据读入、处理、训练、测试、评估
"""

import os
import time
import numpy as np
import torch

from operator import add
from functools import reduce
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Datasets.data_normer import DataNormer
from Datasets.file_reader.reader import DataReader
from Datasets.basic_data.image_field import ImageField
from Datasets.basic_data.scalar_data import ScalarData
from Datasets.data_loader.loader import CustomDataset


from Models.NeuralOperators import NeuralOperators

from Utilizes.work_space import WorkSpace
from Utilizes.visual_data import MatplotlibVision

import torch.nn as nn
class Predictor(nn.Module):

    def __init__(self, branch, trunc, infer, field_dim, infer_dim):

        super(Predictor, self).__init__()

        self.branch_net = branch
        self.trunc_net = trunc
        self.infer_net = infer
        self.field_net = nn.Linear(trunc.out_dim, field_dim)

    def forward(self, input, grid, **kwargs):
        """
        forward compute
        :param design: tensor list[(batch_size, ..., operator_dims[0]), (batch_size, ..., operator_dims[1]), ...]
        :param coords: (batch_size, ..., input_dim)
        """

        T = self.trunc_net(grid)
        B = self.branch_net(input)
        T_size = T.shape[1:-1]
        for i in range(len(T_size)):
            B = B.unsqueeze(1)
        B = torch.tile(B, [1, ] + list(T_size) + [1, ])
        feature = B * T
        F = self.field_net(feature)
        Y = self.infer_net(feature)
        return F


if __name__ == '__main__':
    ################################################################
    # 读取数据
    ################################################################

    data_path = os.path.join('data', 'dim_pro8_single_try.mat')
    reader = DataReader(data_path)
    design = reader.read_data('data')
    fields = reader.read_data('field')
    coords = reader.read_data('grids')
    target = torch.cat((reader.read_data('Nu'), reader.read_data('f')), dim=1)

    design = ScalarData(data=design)
    fields = ImageField(data=fields, grid=coords[..., :2], name=['p', 't', 'u', 'v'])
    target = ScalarData(data=target, name=['Nu', 'f'])

    design_normer = DataNormer(design.data, method='min-max')
    coords_normer = DataNormer(fields.grid, method='min-max')
    fields_normer = DataNormer(fields.data, method='mean-std')
    target_normer = DataNormer(target.data, method='mean-std')
    data_normer = {'input': design_normer, 'output': fields_normer, 'grid': coords_normer, 'target': target_normer}

    train_size = 5000
    valid_size = 1000

    train_dataset = CustomDataset(design[:train_size], fields[:train_size], target=target[:train_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=False)

    valid_dataset = CustomDataset(design[-valid_size:], fields[-valid_size:], target=target[-valid_size:])
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, drop_last=False)

    ################################################################
    # 模型定义
    ################################################################

    from fno.FNOs import FNO2d
    from basic.basic_layers import FcnSingle
    from cnn.ConvNets import DownSampleNet2d

    # 将控制台的结果输出到log文件
    work_path = os.path.dirname(__file__)
    work_name = 'FNO'
    work_space = WorkSpace(work_name=work_name, base_path=work_path)

    if torch.cuda.is_available():
        use_device = torch.device('cuda')
    else:
        use_device = torch.device('cpu')

    FNO_model = FNO2d(in_dim=2, out_dim=64, modes=(16, 16), width=64, depth=4, padding=5, activation='gelu')
    MLP_model = FcnSingle(planes=(design.proper_size, 64, 64, 64), last_activation=True)
    CNN_model = DownSampleNet2d(in_sizes=tuple(fields.grid_shape) + (64,), out_sizes=target.proper_size,
                                width=64, depth=4, dropout=0.05)

    predictor = Predictor(trunc=FNO_model, branch=MLP_model, infer=CNN_model,
                          field_dim=fields.proper_size, infer_dim=target.proper_size)

    work_space.logger.info("Model Name: {:s}, Computing Device: {:s}".format(work_name, str(use_device)))

    ################################################################
    # 训练参数
    ################################################################
    learning_rate = 0.001
    learning_beta = (0.7, 0.9)
    weight_decay = 0.0
    scheduler_step = (100, 150, 200)
    scheduler_gamma = 0.1
    # 损失函数
    loss_func = nn.MSELoss()
    # L1loss = nn.SmoothL1Loss()
    # Loss_metirc = FieldsLpLoss(d=2, p=2, reduction=True, size_average=False)
    # 优化算法
    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate,
                                 betas=learning_beta, weight_decay=weight_decay)
    # 下降策略
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_step, gamma=scheduler_gamma)
    # 可视化

    Visual = MatplotlibVision(work_space.train_path,
                              input_name=('x', 'y'), field_name=fields.name, target_name=target.name)

    module = NeuralOperators(network=predictor, device=use_device,
                             lossfunc=loss_func, optimizer=optimizer, scheduler=scheduler,
                             work_space=work_space, data_normer=data_normer,
                                   )


    ################################################################
    # Train
    ################################################################

    for epoch in range(30):

        module.train(train_loader, valid_loader, epochs=20)

        if epoch > 0 and epoch % 5 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.array(module.history_loss['train']).mean(axis=-1), 'train_step')
            Visual.plot_loss(fig, axs, np.array(module.history_loss['valid']).mean(axis=-1), 'valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(module.work_space.train_path, 'log_loss.svg'))
            plt.close(fig)

        ################################################################
        # Visualization
        ################################################################

        if epoch > 0 and epoch % 5 == 0:

            visual_batch = 10
            train_design, train_true, train_target = train_dataset[:visual_batch]
            valid_design, valid_true, valid_target = valid_dataset[:visual_batch]

            train_pred = module.infer(train_design.data, grid=train_true.grid)
            valid_pred = module.infer(valid_design.data, grid=valid_true.grid)

            for fig_id in range(visual_batch):
                fig, axs = plt.subplots(4, 3, figsize=(20, 10), layout='constrained', num=2)
                Visual.plot_fields_ms(fig, axs, train_true.data[fig_id].numpy(),
                                      train_pred[fig_id],
                                      train_true.grid[fig_id].numpy())
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(visual_batch):
                fig, axs = plt.subplots(4, 3, figsize=(20, 10), layout='constrained', num=3)
                Visual.plot_fields_ms(fig, axs, valid_true.data[fig_id].numpy(),
                                      valid_pred[fig_id],
                                      valid_true.grid[fig_id].numpy())
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

