#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/26 3:16
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : UNets.py
"""
import math
import torch
import torch.nn as nn
from Models.conv_layers import *
from Models.basic_layers import *


class UpSampleNet2d(nn.Module):

    def __init__(self, in_sizes: int, out_sizes: tuple[int, ...], hidden_sizes=(3, 32, 32), conv_width=32, conv_depth=4,
                 activation='gelu',  dropout=0.0):
        """
        :param in_sizes: C_in, int
        :param out_sizes: (C_out, H_out, W_out)

        :param hidden_sizes: hidden (C, H, W)
        :param width: hidden dim, int
        :param depth: hidden layers, int
        """
        super(UpSampleNet2d, self).__init__()
        if not isinstance(hidden_sizes, list):
            hidden_sizes = list(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.out_sizes = out_sizes
        self.width = conv_width
        self.depth = conv_depth
        self.in_dim = in_sizes

        self.linear = nn.Linear(self.in_dim, math.prod(self.hidden_sizes))
        self.conv = UNet2d(in_sizes=hidden_sizes, out_sizes=self.out_sizes, width=self.width, depth=self.depth,
                           activation=activation, dropout=dropout)

    def forward(self, x):
        x = self.linear(x)
        x = x.view([-1, ] + self.hidden_sizes)
        x = self.conv(x)
        return x


class DownSampleNet2d(nn.Module):

    def __init__(self, in_sizes: tuple[int, ...], out_sizes: int, hidden_sizes=(3, 32, 32), conv_width=32, conv_depth=4,
                 activation='gelu',
                 dropout=0.0):
        """
        :param in_sizes: (C_in, H_in, W_in)
        :param out_sizes: C_out, int

        :param hidden_sizes: hidden (C, H, W)
        :param width: hidden dim, int
        :param depth: hidden layers, int
        """
        super(DownSampleNet2d, self).__init__()

        self.in_sizes = in_sizes
        if not isinstance(hidden_sizes, list):
            hidden_sizes = list(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.width = conv_width
        self.depth = conv_depth
        self.out_dim = out_sizes

        self.linear = nn.Linear(math.prod(self.hidden_sizes), self.out_dim)
        self.conv = UNet2d(in_sizes=self.in_sizes, out_sizes=self.hidden_sizes, width=self.width, depth=self.depth,
                           activation=activation, dropout=dropout)

    def forward(self, x):
        x = self.conv(x)
        x = x.view([-1, math.prod(self.hidden_sizes)])
        x = self.linear(x)
        return x


class UNet2d(nn.Module):

    def __init__(self, in_sizes: tuple[int, ...], out_sizes: tuple[int, ...], width=32, depth=4, activation='gelu',
                 dropout=0.0):
        """
        :param in_sizes: (C_in, H_in, W_in)
        :param out_sizes: (C_out, H_out, W_out)
        :param width: hidden dim, int
        :param depth: hidden layers, int
        """
        super(UNet2d, self).__init__()

        self.in_sizes = in_sizes[1:]
        self.out_sizes = out_sizes[1:]
        self.in_dim = in_sizes[0]
        self.out_dim = out_sizes[0]
        self.width = width
        self.depth = depth

        self._input_sizes = [0, 0]
        self._input_sizes[0] = max(2 ** math.floor(math.log2(in_sizes[0])), 2 ** depth)
        self._input_sizes[1] = max(2 ** math.floor(math.log2(in_sizes[1])), 2 ** depth)

        self.interp_in = Interp2dUpsample(in_dim=self.in_dim, out_dim=self.in_dim, activation=activation,
                                          dropout=dropout,
                                          interp_size=self._input_sizes, conv_block=False)
        self.encoders = nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
                self.encoders.append(
                    Conv2dResBlock(self.in_dim, width, basic_block=True, activation=activation, dropout=dropout))
            else:
                self.encoders.append(nn.Sequential(nn.MaxPool2d(2, 2),
                                                   Conv2dResBlock(2 ** (i - 1) * width, 2 ** i * width,
                                                                  basic_block=True, activation=activation,
                                                                  dropout=dropout)))

        self.bottleneck = nn.Sequential(nn.MaxPool2d(2, 2),
                                        Conv2dResBlock(2 ** i * width, 2 ** i * width * 2, basic_block=True,
                                                       activation=activation, dropout=dropout))

        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        for i in range(self.depth, 0, -1):
            self.decoders.append(
                Conv2dResBlock(2 ** i * width, 2 ** (i - 1) * width, activation=activation, dropout=dropout))
            self.upconvs.append(
                DeConv2dBlock(2 ** i * width, 2 ** (i - 1) * width, 2 ** (i - 1) * width, activation=activation,
                              dropout=dropout))

        self.conv1 = Conv2dResBlock(in_dim=width, out_dim=self.out_dim, basic_block=False, activation=activation,
                                    dropout=dropout)

        self.interp_out = Interp2dUpsample(in_dim=self.out_dim, out_dim=self.out_dim, interp_size=self.out_sizes,
                                           conv_block=False, activation=activation, dropout=dropout)

        self.conv2 = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        enc = []
        enc.append(self.interp_in(x))
        for i in range(self.depth):
            enc.append(self.encoders[i](enc[-1]))

        x = self.bottleneck(enc[-1])

        for i in range(self.depth):
            x = self.upconvs[i](x)
            x = torch.cat((x, enc[-i - 1]), dim=1)
            x = self.decoders[i](x)

        x = self.interp_out(self.conv1(x))
        x = self.conv2(x)
        return x


if __name__ == '__main__':
    x = torch.ones([10, 4, 92, 8])
    input_size = x.shape[1:]
    layer = UNet2d(in_sizes=input_size, out_sizes=(5, 32, 32), width=32, depth=6)
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 10])
    in_sizes, hidden_size, out_sizes = 10, (4, 92, 32), (5, 32, 32)
    layer = UpSampleNet2d(in_sizes, out_sizes, hidden_size, conv_width=32, conv_depth=4)
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 4, 92, 8])
    input_size, out_sizes, hidden_sizes = (4, 92, 8), 10, (5, 32, 32)
    layer = DownSampleNet2d(input_size, out_sizes, hidden_sizes, conv_width=32, conv_depth=4)
    y = layer(x)
    print(y.shape)
