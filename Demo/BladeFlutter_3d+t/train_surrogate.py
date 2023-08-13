#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/7/9 23:07
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : optim_FNO.py
"""
import numpy as np
import torch
import torch.nn as nn
from Utilizes.process_data import MatLoader
from basic.basic_layers import FcnSingle
from Utilizes.process_data import DataNormer
from Utilizes.visual_data import MatplotlibVision, TextLogger

import matplotlib.pyplot as plt
import time
import os
from torchinfo import summary
from shutil import copyfile
from operator import add
from functools import reduce

from utilize import predictor, train, valid, inference, data_preprocess, custom_dataset, cal_damps

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":

    for train_size in (4, 7, 10):

        ################################################################
        # configs
        ################################################################

        name = 'MLP_' + str(train_size)
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

        epochs = 101
        learning_rate = 0.001
        scheduler_step = 51
        scheduler_gamma = 0.1

        Logger.info('Total epochs: {:d}, learning_rate: {:e}, scheduler_step: {:d}, scheduler_gamma: {:.3e}'
                    .format(epochs, learning_rate, scheduler_step, scheduler_gamma))

    ################################################################
    # load data
    ################################################################
        data_path = 'data/y_bend-10_RN1_A5_APhis_trans_field.mat'

        reader = MatLoader(data_path)
        design, _, _, target, index = data_preprocess(reader, train_size=train_size)
        train_index, valid_index = index
        damper = torch.tensor(cal_damps(target).min(axis=-1, keepdims=True), dtype=torch.float32)

        design_normer = DataNormer(design)
        damper_normer = DataNormer(damper, method='mean-std')
        design = design_normer.norm(design)
        damper = damper_normer.norm(damper)


        plt.figure(10)
        plt.clf()
        plt.scatter(design_normer.back(design),
                    cal_damps(target).min(axis=-1), label='all')
        plt.scatter(design_normer.back(design[train_index]),
                    cal_damps(target[train_index]).min(axis=-1), label='train')
        plt.scatter(design_normer.back(design[valid_index]),
                    cal_damps(target[valid_index]).min(axis=-1), label='valid')
        plt.legend()
        plt.savefig(os.path.join(work_path, 'train_test_split.jpg'))

        del reader

        Net_model = FcnSingle(planes=(1, 64, 64, 64, 1), last_activation=False).to(Device)

        # 损失函数
        Loss_func = nn.MSELoss()
        # 优化算法
        Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=0.0)
        # 下降策略
        Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        # 可视化
        field_name = reduce(add, [['P_' + str(i), 'M_' + str(i)] for i in range(1, 6)])
        Visual = MatplotlibVision(train_path, input_name=('x', 'y'), field_name=field_name)

        ################################################################
        Log_loss = []
        for epoch in range(epochs):

            train_pred = Net_model(design[train_index].to(Device))
            train_loss = Loss_func(train_pred, damper[train_index].to(Device))
            Optimizer.zero_grad()
            train_loss.backward()
            Optimizer.step()

            if epoch > scheduler_step:
                Scheduler.step()

            valid_pred = Net_model(design[valid_index].to(Device))
            valid_loss = Loss_func(valid_pred, damper[valid_index].to(Device))

            Log_loss.append([train_loss.item(), valid_loss.item()])

            print(Log_loss[-1])

        train_pred = damper_normer.back(train_pred.detach().cpu())
        valid_pred = damper_normer.back(valid_pred.detach().cpu())

        plt.figure(10)
        plt.clf()
        plt.scatter(design_normer.back(design[train_index]), train_pred, label='train')
        plt.scatter(design_normer.back(design[valid_index]), valid_pred, label='valid')
        plt.legend()
        plt.savefig(os.path.join(work_path, 'result.jpg'))

        from Tasks.Optimizer import TaskOptimizer
        import datetime
        def get_obj(x):
            with torch.no_grad():
                x = torch.tensor(design_normer.norm(x), dtype=torch.float32).to(Device)
                obj_pred = damper_normer.back(Net_model(x))
            return -obj_pred.cpu().numpy()

        DampOptimizer = TaskOptimizer(optimal_parameters=['alpha', ], object_function=get_obj, optimizer_name='DE',
                                      lower_bound=[0.0, ], upper_bound=[360., ])
        DampOptimizer._build_optimizer(size_pop=10, max_iter=50)

        start_time = datetime.datetime.now()
        best_x, best_y = DampOptimizer.run()
        end_time = datetime.datetime.now()

        print('best_d_parameters: {}, \n  best_objective_function: {}, costs {:.2f}'.
              format(best_x, -best_y, (end_time - start_time).total_seconds()))




