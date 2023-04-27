#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/2/19 23:59
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : run_pinn_Bayesian_2d.py
"""
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

import Utilizes.util as util
from Utilizes.visual_data import MatplotlibVision, TextLogger
import Bayesian_util as Bayesian
from basic.basic_layers import FcnSingle

import os
import sys

# device

print(f'Is CUDA available?: {torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# hyperparameters

util.set_random_seed(123)
prior_std = 1
like_std = 0.1
step_size = 0.001
burn = 200
num_samples = 400
L = 100
layer_planes = [2, 16, 16, 1]
activation = torch.tanh
pde = True
pinns = False
epochs = 10000
tau_priors = 1 / prior_std ** 2
tau_likes = 1 / like_std ** 2

lb = -1
ub = 1
N_tr_f = 500
N_tr_b = 25
N_val = 100


# data

def u(x):
    return torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])


def f(x):
    return 0.01 * -2 * np.pi ** 2 * u(x) + u(x) * (u(x) ** 2 - 1)


data = {}
xb = torch.linspace(lb, ub, N_tr_b)
xb = torch.cartesian_prod(xb, xb)
xb = xb[torch.sum((xb == 1) + (xb == -1), 1).bool(), :]
data['x_u'] = xb
data['y_u'] = u(data['x_u']) + torch.randn_like(u(data['x_u'])) * like_std
data['x_f'] = (ub - lb) * torch.rand(N_tr_f, 2) + lb
data['y_f'] = f(data['x_f']) + torch.randn_like(f(data['x_f'])) * like_std

data_val = {}
xu = torch.linspace(lb, ub, N_val)
data_val['x_u'] = torch.cartesian_prod(xu, xu)
data_val['y_u'] = u(data_val['x_u'])
data_val['x_f'] = torch.cartesian_prod(xu, xu)
data_val['y_f'] = f(data_val['x_f'])

for d in data:
    data[d] = data[d].to(device)
for d in data_val:
    data_val[d] = data_val[d].to(device)

# model

name = '2d'
work_path = os.path.join('work', name)
isCreated = os.path.exists(work_path)
if not isCreated:
    os.makedirs(work_path)

# 将控制台的结果输出到a.log文件，可以改成a.txt
sys.stdout = TextLogger(os.path.join(work_path, 'train.log'), sys.stdout)

net_u = FcnSingle(layer_planes, activation='tanh').to(device)
nets = [net_u]


def model_loss(data, fmodel, params_unflattened, tau_likes, gradients, params_single=None):
    x_u = data['x_u']
    y_u = data['y_u']
    pred_u = fmodel[0](x_u, params=params_unflattened[0])
    ll = - 0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
    x_f = data['x_f']
    x_f = x_f.detach().requires_grad_()
    u = fmodel[0](x_f, params=params_unflattened[0])
    Du = gradients(u, x_f)[0]
    u_x, u_y = Du[:, 0:1], Du[:, 1:2]
    u_xx = gradients(u_x, x_f)[0][:, 0:1]
    u_yy = gradients(u_y, x_f)[0][:, 1:2]
    pred_f = 0.01 * (u_xx + u_yy) + u * (u ** 2 - 1)
    y_f = data['y_f']
    ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
    output = [pred_u, pred_f]

    if torch.cuda.is_available():
        del x_u, y_u, x_f, y_f, u, u_x, u_y, u_xx, u_yy, pred_u, pred_f
        torch.cuda.empty_cache()

    return ll, output


# sampling

params_hmc = Bayesian.sample_model_bpinns(nets, data, model_loss=model_loss, num_samples=num_samples,
                                          num_steps_per_sample=L, step_size=step_size, burn=burn, tau_priors=tau_priors,
                                          tau_likes=tau_likes, device=device, pde=pde, pinns=pinns, epochs=epochs)

pred_list, log_prob_list = Bayesian.predict_model_bpinns(nets, params_hmc, data_val, model_loss=model_loss,
                                                         tau_priors=tau_priors, tau_likes=tau_likes, pde=pde)

print('\nExpected validation log probability: {:.3f}'.format(torch.stack(log_prob_list).mean()))

pred_list_u = pred_list[0].cpu().numpy()
pred_list_f = pred_list[1].cpu().numpy()

# plot

extent = [lb, ub, lb, ub]
u_val = data_val['y_u'].cpu().numpy()
x_f = data['x_f'].cpu().numpy()

pred_mean_u = pred_list_u.mean(0).reshape(N_val, N_val)
error = abs(pred_mean_u - u_val.reshape(N_val, N_val))
plt.figure(figsize=(5, 5))
plt.imshow(error, extent=extent)
plt.colorbar()
plt.plot(x_f[:, 0], x_f[:, 1], 'kx', markersize=3)
# plt.show()
plt.savefig(os.path.join(work_path, 'mean_u.jpg'))

twostd = 2 * pred_list_u.std(0).reshape(N_val, N_val)
plt.figure(figsize=(5, 5))
plt.imshow(twostd, extent=extent)
plt.colorbar()
plt.plot(x_f[:, 0], x_f[:, 1], 'kx', markersize=3)
plt.savefig(os.path.join(work_path, 'std_u.jpg'))
# plt.show()
# %%
