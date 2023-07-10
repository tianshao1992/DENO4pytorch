#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/7/9 20:32
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : valid_FNO.py
"""

import numpy as np
import torch
import torch.nn as nn
from Utilizes.process_data import MatLoader
from fno.FNOs import FNO2d
from basic.basic_layers import FcnSingle
from cnn.ConvNets import DownSampleNet2d
from Utilizes.process_data import DataNormer
from Utilizes.visual_data import MatplotlibVision, TextLogger

import datetime
import matplotlib.pyplot as plt
import os
from operator import add
from functools import reduce, partial

from utilize import predictor, data_preprocess, cal_damps


import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def obj_func(d, c, fields_return=False):
    """
        :param x: (1,)
    """
    with torch.no_grad():
        d = torch.from_numpy(d).float().to(c)
        d = d[None, :].repeat([c.shape[0], 1])
        d = design_normer.norm(d)
        f_p, t_p = Net_model(d, c)
        # d = design_normer.back(d).cpu().numpy()
        t_p = target_normer.back(t_p).cpu()
        y_p = -cal_damps(t_p.unsqueeze(0)).squeeze(0).min().item()

    if fields_return:
        return y_p, f_p, t_p
    else:
        return y_p


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'FNO+MLP+CNN-1'
    work_path = os.path.join('work', name)
    train_path = os.path.join(work_path, 'train')
    infer_path = os.path.join(work_path, 'infer')
    isCreated = os.path.exists(infer_path)
    if not isCreated:
        os.makedirs(infer_path)

    # 将控制台的结果输出到log文件
    Logger = TextLogger(os.path.join(infer_path, 'valid.log'))

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')
    Logger.info("Model Name: {:s}, Computing Device: {:s}".format(name, str(Device)))

    ntrain = 20
    nvalid = 5
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

    ################################################################
    # Neural Networks
    ################################################################

    FNO_model = FNO2d(in_dim=20, out_dim=64, modes=(16, 16), width=64, depth=4,
                      padding=5, activation='gelu').to(Device)
    MLP_model = FcnSingle(planes=(1, 64, 64, 64), last_activation=True).to(Device)
    CNN_model = DownSampleNet2d(in_sizes=tuple(fields.shape[1:-1]) + (64,), out_sizes=target.shape[-1],
                                width=64, depth=4)

    Net_model = predictor(trunc=FNO_model, branch=MLP_model, infer=CNN_model,
                          field_dim=10, infer_dim=target.shape[-1]).to(Device)

    try:
        checkpoint = torch.load(os.path.join(train_path, 'best_model.pth'))
        Net_model.load_state_dict(checkpoint['net_model'])
        Logger.warning("model load successful!")
    except:
        Logger.warning("model doesn't exist!")

    # 可视化
    field_name = reduce(add, [['P_' + str(i), 'M_' + str(i)] for i in range(1, 6)])
    Visual = MatplotlibVision(infer_path, input_name=('x', 'y'), field_name=field_name)

    ################################################################
    # infer process
    ################################################################

    Net_model.eval()

    log_x = []
    log_y = []
    with torch.no_grad():
        for i in range(len(design)):

            d, c, f_t, t_t = design[i].to(Device), coords[i].to(Device), fields[i].to(Device), target[i].to(Device)
            d = d.unsqueeze(0).repeat([c.shape[0], 1])
            f_p, t_p = Net_model(d, c)

            d = design_normer.back(d).cpu().numpy()
            f_t, f_p = fields_normer.back(f_t), fields_normer.back(f_p).cpu()
            t_t, t_p = target_normer.back(t_t), target_normer.back(t_p).cpu()

            y_t = cal_damps(t_t.unsqueeze(0)).squeeze(0)
            y_p = cal_damps(t_p.unsqueeze(0)).squeeze(0)

            log_x.append(float(d[0]))
            log_y.append([float(y_t.min()), float(y_p.min())])

            fig, axs = plt.subplots(1, 2, figsize=(20, 10), num=1)

            Visual.plot_fields1d(fig, axs, y_t[:, None], y_p[:, None], coord=np.arange(y_t.shape[0]), show_channel=(0,),
                                 xylabels=('叶片编号', '气动阻尼'), legends=('真实值', '预测值', '偏差'))

            fig.suptitle('相位{:.1f}: 实际最小值{:.2e}，预测最小值{:.2e}，相对偏差{:.2f}%'
                         .format(float(d[0]), y_t.min(), y_p.min(), (y_p.min() - y_t.min()) / y_t.min() * 100),
                         font=Visual.font)
            fig.savefig(os.path.join(infer_path, 'damps_prediction{:.1f}.jpg'.format(float(d[0]))))
            plt.close(fig)
            #


    log_x, log_y = np.array(log_x), np.array(log_y)
    index = np.argsort(log_x)
    log_x, log_y = log_x[index], log_y[index]

    fig, axs = plt.subplots(1, 1, figsize=(20, 10), num=1)
    Visual.plot_value(fig, axs, log_x, log_y[:, 0], label='真实最小气动阻尼',
                      xylabels=('相位', '气动阻尼'))
    Visual.plot_value(fig, axs, log_x, log_y[:, 1], label='预测最小气动阻尼',
                      xylabels=('相位', '气动阻尼'))
    fig.savefig(os.path.join(infer_path, 'min_damps_prediction.jpg'))
    plt.close(fig)

    ################################################################
    # optim process
    ################################################################
    general_coords = coords[0].to(Device)

    from Tasks.Optimizer import TaskOptimizer

    get_obj = partial(obj_func, c=general_coords, fields_return=False)
    DampOptimizer = TaskOptimizer(optimal_parameters=['alpha',], object_function=get_obj, optimizer_name='DE',
                                  lower_bound=[0.0,], upper_bound=[360.,])

    DampOptimizer._build_optimizer(size_pop=10, max_iter=50)

    start_time = datetime.datetime.now()
    best_x, best_y = DampOptimizer.run()

    end_time = datetime.datetime.now()
    print('best_d_parameters: {}, \n  best_objective_function: {}, costs {:.2f}'.
          format(best_x, -best_y, (end_time - start_time).total_seconds()))

    obj_pred, f_pred, t_pred = obj_func(best_x, general_coords, fields_return=True)
