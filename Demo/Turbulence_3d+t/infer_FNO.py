# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/6/19 14:56
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：inference_FNO.py
@File ：inference_FNO.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
from fno.FNOs import FNO3d

from Utilizes.visual_data import MatplotlibVision, TextLogger
from Utilizes.process_data import DataNormer
from Utilizes.loss_metrics import FieldsLpLoss

import matplotlib.pyplot as plt
import time

from train_FNO import feature_transform, custom_dataset

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
    Logger = TextLogger(os.path.join(work_path, 'inference.log'))
    #  torch.cuda.set_device(1)

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')
    Logger.info("Model Name: {:s}, Computing Device: {:s}".format(name, str(Device)))

    in_dim = 3
    out_dim = 3
    infer_T = 200
    ntrain = 40
    ninfer = 10

    mode = 16
    modes = (mode, mode, mode)
    width = 64
    depth = 4
    steps = 5
    padding = 0
    dropout = 0.0

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
    infer_data = all_data[-ninfer:, :, ::r1, ::r2, ::r3][..., :s1, :s2, :s3, :]

    ################################################################
    # Neural Networks
    ################################################################

    # 建立网络

    Net_model = FNO3d(in_dim=in_dim, out_dim=out_dim, modes=modes, width=width, depth=depth, steps=steps,
                      padding=padding, activation='gelu').to(Device)

    try:
        checkpoint = torch.load(os.path.join(work_path, 'latest_model.pth'))
        Net_model.load_state_dict(checkpoint['net_model'])
        Logger.warning("model load successful!")
    except:
        Logger.warning("model doesn't exist!")

    input1 = torch.randn(16, 32, 32, 32, 5 * 3).to(Device)
    input2 = torch.randn(16, 32, 32, 32, 3).to(Device)
    model_statistics = summary(Net_model, input_data=[input1, input2], device=Device, verbose=0)
    Logger.write(str(model_statistics))

    # 损失函数
    Loss_metirc = FieldsLpLoss(d=2, p=2, reduction=True, size_average=False)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y', 'z'), field_name=('u', 'v', 'w'))

    star_time = time.time()
    log_loss = [[], []]
    log_per = [[], []]

    ################################################################
    # infer process
    ################################################################

    initial_input = infer_data[:, :steps]

    preds = []
    truth = infer_data[:, steps:infer_T + steps]

    with torch.no_grad():
        for t in range(infer_T):

            if t == 0:
                x = torch.tensor(initial_input.transpose((0, 2, 3, 4, 5, 1)), dtype=torch.float32).to(Device)
                input_sizes = list(x.shape)
            else:
                x = torch.cat((x[..., -steps + 1:], yy), dim=-1)

            xx = x.reshape(input_sizes[:-2] + [-1, ])
            gd = feature_transform(xx)
            yy = Net_model(xx, gd).unsqueeze(-1)
            preds.append(yy.cpu().numpy())

    preds = np.concatenate(preds, axis=-1).transpose((0, 5, 1, 2, 3, 4))

    error = []
    for t in range(infer_T):
        error.append(Loss_metirc(preds[:, t].reshape([input_sizes[0], -1, 1]),
                                 truth[:, t].reshape([input_sizes[0], -1, 1])))

    error = np.stack(error, axis=1)

    avg_error = np.mean(error, axis=0)
    std_error = np.std(error, axis=0)

    fig, axs = plt.subplots(1, 1, figsize=(10, 5), num=2, constrained_layout=True)
    Visual.plot_value(fig, axs, np.arange(infer_T), avg_error, std=std_error, label='relative_l2_loss',
                      xylabels=('time_steps', 'lp_loss'))
    fig.savefig(os.path.join(work_path, '{}.svg'.format('time_lploss')))
    plt.close(fig)

    np.savetxt(os.path.join(work_path, '{}.txt'.format('time_lploss')), np.concatenate((avg_error, std_error), axis=-1))
