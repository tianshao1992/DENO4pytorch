#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/7/23 10:40
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : run_demo.py
# @Description    : ******
"""

import numpy as np
import torch
import torch.nn as nn
import os

import matplotlib.pyplot as plt

from Utilizes.process_data import DataNormer, MatLoader
from Utilizes.visual_data import MatplotlibVision


file_path = os.path.join('data', 'dim_pro8_single_try.mat')
reader = MatLoader(file_path)

fields = reader.read_field('field')
design = reader.read_field('data')
coords = reader.read_field('grids')[..., :2]
target = torch.concat((reader.read_field('Nu'), reader.read_field('f')), dim=-1)

Visual = MatplotlibVision(os.path.join('work', 'visual'), input_name=('x', 'y'), field_name=('p', 't', 'u', 'v'))

# fig, axs = plt.subplots(4, 3, figsize=(30, 6), num=1)
# Visual.plot_fields_ms(fig, axs, real=fields[0].numpy(), pred=fields[0].numpy(), coord=coords[0].numpy())
# plt.show()

from Models.basic.basic_layers import FcnSingle, FcnMulti

design_shape = design.shape[-1]
fields_shape = fields.shape[-1]
coords_shape = coords.shape[-1]
target_shape = target.shape[-1]

design_tile = torch.tile(design[:, None, None, :], (1, 792, 40, 1))
x = torch.concat((design_tile, coords), dim=-1)
layer = FcnSingle([design_shape + coords_shape, 64, 64, fields_shape])
y = layer(x)
print('fcn single output: ', y.shape)

layer = FcnMulti([design_shape + coords_shape, 64, 64, 4])
y = layer(x)
print('fcn multi output: ', y.shape)

from Models.don.DeepONets import DeepONetMulti

# design_ = [design_tile, ]
# layer = DeepONetMulti(input_dim=coords_shape, operator_dims=[design_shape, ], output_dim=fields_shape,
#                       planes_branch=[64] * 3, planes_trunk=[64] * 2)
# y = layer(design_, coords, size_set=False)
# print('deep_onets output: ', y.shape)


from Models.cnn.ConvNets import UNet2d, UpSampleNet2d, DownSampleNet2d

input_size = design_tile.shape[1:]
layer = UNet2d(in_sizes=input_size, out_sizes=fields.shape[1:], width=32, depth=6)
y = layer(design_tile[:, :, :, :], coords[:, :, :, :])
print('UNet2d: ', y.shape)

layer = UpSampleNet2d(design_shape, out_sizes=fields.shape[1:], width=32, depth=4)
y = layer(design)
print('UpSampleNet2d output: ', y.shape)

layer = DownSampleNet2d(in_sizes=fields.shape[1:], out_sizes=design_shape, width=32, depth=4)
y = layer(fields)
print('DownSampleNet2d output: ', y.shape)


from Models.fno.FNOs import FNO2d

input_size = design_tile.shape[1:]
layer = FNO2d(in_dim=design_shape, out_dim=fields_shape, width=64, depth=4)
y = layer(design_tile, coords)
print('fno 2d output: ', y.shape)

y = layer(design_tile)
print('fno 2d output: ', y.shape)

