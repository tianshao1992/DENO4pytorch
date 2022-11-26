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

    def __init__(self, in_sizes: int, out_sizes: tuple[int, ...], width=32, depth=4,
                 activation='gelu', dropout=0.0):
        """
        :param in_sizes: C_in, int
        :param out_sizes: (C_out, H_out, W_out)
        :param width: hidden dim, int
        :param depth: hidden layers, int maybe adjust based on the in_sizes
        :param activation: str 'gelu' 'relu' 'silu' 'tanh'
        :param dropout: dropout, float
        """
        super(UpSampleNet2d, self).__init__()
        self.out_sizes = out_sizes
        self.width = width
        self.depth = depth
        self.in_dim = in_sizes
        self.dropout = dropout
        self.depth = min(math.floor(math.log2(out_sizes[1])) - 1, math.floor(math.log2(out_sizes[2])) - 1, depth)
        self.hidden_size = [0, 0]
        self.hidden_size[0] = 2 ** math.floor(math.log2(out_sizes[1] / 2 ** self.depth))
        self.hidden_size[1] = 2 ** math.floor(math.log2(out_sizes[2] / 2 ** self.depth))

        self.linear = nn.Linear(self.in_dim, math.prod(self.hidden_size) * self.width)

        self.upconvs = nn.ModuleList()
        for i in range(self.depth):
            self.upconvs.append(
                Interp2dUpsample(self.width, self.width, residual=True, conv_block=True,
                                 interp_mode='bilinear', activation=activation, dropout=self.dropout,
                                 interp_size=(self.hidden_size[0] * 2 ** (i + 1), self.hidden_size[1] * 2 ** (i + 1)), )
            )
        self.interp_out = Interp2dUpsample(in_dim=width, out_dim=self.out_sizes[0], residual=False, conv_block=True,
                                           interp_mode='bilinear', activation=activation, dropout=self.dropout,
                                           interp_size=self.out_sizes[1:], )
        self.conv = nn.Conv2d(self.out_sizes[0], self.out_sizes[0], kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, x):
        x = self.linear(x)
        x = x.view([-1, self.width] + self.hidden_size)
        for i in range(self.depth):
            x = self.upconvs[i](x)
        x = self.interp_out(x)
        x = self.conv(x)
        return x


class DownSampleNet2d(nn.Module):

    def __init__(self, in_sizes: tuple[int, ...], out_sizes: int, width=32, depth=4, activation='gelu', dropout=0.0):
        """
        :param in_sizes: (C_in, H_in, W_in)
        :param out_sizes: C_out, int
        :param width: hidden dim, int
        :param depth: hidden layers, int maybe adjust based on the in_sizes
        :param activation: str 'gelu' 'relu' 'silu' 'tanh'
        :param dropout: dropout, float
        """
        super(DownSampleNet2d, self).__init__()

        self.in_sizes = in_sizes
        self.width = width
        self.depth = depth
        self.out_dim = out_sizes
        self.dropout = dropout
        self.activation = activation
        log2_in = [math.floor(math.log2(in_sizes[1])), math.floor(math.log2(in_sizes[2]))]
        self.depth = min(log2_in[0] - 1, log2_in[1] - 1, depth)
        self._out_size = [2 ** (log2_in[0] - self.depth), 2 ** (log2_in[1] - self.depth)]

        self.interp_in = Interp2dUpsample(in_dim=self.in_sizes[0], out_dim=self.width, residual=False, conv_block=True,
                                          interp_mode='bilinear', activation=self.activation, dropout=self.dropout,
                                          interp_size=(2 ** log2_in[0], 2 ** log2_in[1]))
        self.downconvs = nn.ModuleList()
        for i in range(self.depth):
            self.downconvs.append(nn.Sequential(
                Conv2dResBlock(self.width, self.width, basic_block=True, activation=activation, dropout=dropout),
                nn.AvgPool2d(2, 2), ))

        self.linear = nn.Sequential(nn.Linear(math.prod(self._out_size) * self.width, 64),
                                    activation_dict[activation],
                                    nn.Linear(64, self.out_dim)
                                    )

    def forward(self, x):
        x = self.interp_in(x)
        for i in range(self.depth):
            x = self.downconvs[i](x)
        x = x.view([-1, math.prod(self._out_size) * self.width])
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
    # x = torch.ones([10, 4, 92, 8])
    # input_size = x.shape[1:]
    # layer = UNet2d(in_sizes=input_size, out_sizes=(5, 32, 32), width=32, depth=6)
    # y = layer(x)
    # print(y.shape)

    # x = torch.ones([10, 10])
    # in_sizes, out_sizes = 10, (5, 58, 32)
    # layer = UpSampleNet2d(in_sizes, out_sizes, width=32, depth=4)
    # y = layer(x)
    # print(y.shape)

    x = torch.ones([10, 4, 92, 52])
    in_sizes, out_sizes = x.shape[1:], 10
    layer = DownSampleNet2d(in_sizes, out_sizes, width=32, depth=4)
    y = layer(x)
    print(y.shape)
