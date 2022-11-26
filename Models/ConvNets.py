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





class UNet2d(nn.Module):

    def __init__(self, in_sizes, out_sizes, width=32, depth=4):
        """
        :param in_sizes: (H_in, W_in, C_in)
        :param out_sizes: (H_out, W_out, C_out)
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

        self.interp_in = Interp2dUpsample(in_dim=self.in_dim, out_dim=self.in_dim,
                                          interp_size=self._input_sizes, conv_block=False)
        self.encoders = nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
                self.encoders.append(Conv2dResBlock(self.in_dim, width, basic_block=True))
            else:
                self.encoders.append(nn.Sequential(nn.MaxPool2d(2, 2),
                                                   Conv2dResBlock(2 ** (i - 1) * width, 2 ** i * width,
                                                                  basic_block=True)))

        self.bottleneck = nn.Sequential(nn.MaxPool2d(2, 2),
                                        Conv2dResBlock(2 ** i * width, 2 ** i * width * 2, basic_block=True))

        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        for i in range(self.depth, 0, -1):
            self.decoders.append(Conv2dResBlock(2 ** i * width, 2 ** (i - 1) * width))
            self.upconvs.append(DeConv2dBlock(2 ** i * width, 2 ** (i - 1) * width, 2 ** (i - 1) * width))

        self.conv1 = Conv2dResBlock(in_dim=width, out_dim=self.out_dim, basic_block=False)

        self.interp_out = Interp2dUpsample(in_dim=self.out_dim, out_dim=self.out_dim, interp_size=self.out_sizes,
                                           conv_block=False)

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

    # x = torch.ones([10, 32, 32, 32, 4])
    # g = torch.ones([10, 32, 32, 32, 3])
    # layer = UNet3d(num_channels=4, modes=(5, 3, 5), steps=1)
    # y = layer(x, g)
    # print(y.shape)
