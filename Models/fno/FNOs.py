#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/26 2:14
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : FNOs.py
"""

from fno.spectral_layers import *


class FNO1d(nn.Module):
    """
        1维FNO网络
    """

    def __init__(self, in_dim, out_dim, modes=16, width=64, depth=4, steps=1, padding=2, activation='gelu'):
        super(FNO1d, self).__init__()
        """
        The overall network. It contains /depth/ layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. /depth/ layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes = modes
        self.width = width
        self.depth = depth
        self.steps = steps
        self.activation = activation
        self.padding = padding  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(steps * in_dim + 1, self.width)  # input channel is 2: (a(x), x)

        self.convs = nn.ModuleList()
        for i in range(self.depth):
            self.convs.append(SpectralConv1d(self.width, self.width, self.modes, activation=self.activation, norm=None))

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x, grid):
        """
        forward computation
        """
        # x dim = [b, x1, t*v]
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        for i in range(self.depth):
            x = self.convs[i](x)

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class FNO2d(nn.Module):
    """
        2维FNO网络
    """

    def __init__(self, in_dim, out_dim, modes=(8, 8), width=32, depth=4, steps=1, padding=2,
                 activation='gelu', dropout=0.0):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x, y, c)
        output: the solution of the next timestep
        output shape: (batchsize, x, y, c)
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes = modes
        self.width = width
        self.depth = depth
        self.steps = steps
        self.padding = padding  # pad the domain if input is non-periodic
        self.activation = activation
        self.dropout = dropout
        self.fc0 = nn.Linear(steps * in_dim + 2, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.convs = nn.ModuleList()
        for i in range(self.depth):
            self.convs.append(SpectralConv2d(self.width, self.width, self.modes, activation=self.activation, dropout=self.dropout, norm=None))

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x, grid):
        """
        forward computation
        """
        # x dim = [b, x1, x2, t*v]
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])  # pad the domain if input is non-periodic

        for i in range(self.depth):
            x = self.convs[i](x)

        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)  # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class FNO2dMultChannel(nn.Module):
    """
        2维FNO网络,多通道
    """
    def __init__(self, in_dim, out_dim, modes=(8, 8), width=32, depth=4, steps=1, padding=2,
                 activation='gelu', dropout=0.0): # 构造函数输入保持不变
        super(FNO2dMultChannel, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x, y, c)
        output: the solution of the next timestep
        output shape: (batchsize, x, y, c)
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes = modes
        self.width = width
        self.depth = depth
        self.steps = steps
        self.padding = padding  # pad the domain if input is non-periodic
        self.activation = activation
        self.dropout = dropout
        self.fc0 = nn.Linear(steps * in_dim + 2, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.convs = nn.ModuleList()
        for _ in range(self.out_dim):
            conv = nn.ModuleList()
            for i in range(self.depth):
                conv.append(
                    SpectralConv2d(self.width, self.width, self.modes, activation=self.activation, dropout=self.dropout,
                                   norm=None))
            self.convs.append(conv)

        self.fc1s = nn.ModuleList()
        for _ in range(self.out_dim):
            fc1 =nn.Linear(self.width, 128)
            self.fc1s.append(fc1)

        self.fc2s = nn.ModuleList()
        for _ in range(self.out_dim):
            fc2 = nn.Linear(128, 1)
            self.fc2s.append(fc2)

    def forward(self, x, grid):
        """
        forward computation
        """
        # x dim = [b, x1, x2, t*v]
        x = torch.cat((x, grid), dim=-1) #连接坐标和输入值
        x = self.fc0(x) # 输入层
        x = x.permute(0, 3, 1, 2) #调整四个通道的顺序

        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])  # pad the domain if input is non-periodic
            # 对张量x进行填充操作

        xList = [x] * self.out_dim
        for channel in range(self.out_dim):
            x_cur = xList[channel]
            for i in range(self.depth):
                x_cur = self.convs[channel][i](x_cur)
            if self.padding != 0:
                x_cur = x_cur[..., :-self.padding, :-self.padding]

            x_cur = x_cur.permute(0, 2, 3, 1)  # pad the domain if input is non-periodic #把四个通道的顺序调整回来
            x_cur = self.fc1s[channel](x_cur)
            x_cur = F.gelu(x_cur)
            x_cur = self.fc2s[channel](x_cur)

            xList[channel] = x_cur

        x = torch.cat(xList, dim=-1)  # 将输出沿着最后一个维度连接
        
        return x

class FNO3d(nn.Module):
    """
        3维FNO网络
    """

    def __init__(self, in_dim, out_dim, modes=(8, 8, 8), width=32, depth=4, steps=1, padding=6, activation='gelu'):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes = modes
        self.width = width
        self.depth = depth
        self.steps = steps
        self.padding = padding  # pad the domain if input is non-periodic
        self.activation = activation
        self.fc0 = nn.Linear(steps * in_dim + 3, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.convs = nn.ModuleList()
        for i in range(self.depth):
            self.convs.append(SpectralConv3d(self.width, self.width, self.modes, activation=self.activation, norm=None))

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x, grid):
        """
        forward computation
        x dim = [b, x1, x2, x3, t*v]
        """
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])  # pad the domain if input is non-periodic

        for i in range(self.depth):
            x = self.convs[i](x)

        x = x[..., :-self.padding, :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    x = torch.ones([10, 32, 4])
    g = torch.ones([10, 32, 1])
    layer = FNO1d(in_dim=4, out_dim=1, modes=16, width=64, depth=4, steps=1, padding=2, activation='gelu')
    y = layer(x, g)
    print(y.shape)

    x = torch.ones([10, 32, 32, 4])
    g = torch.ones([10, 32, 32, 2])
    layer = FNO2d(in_dim=4, out_dim=1, modes=(12, 12), width=64, depth=4, steps=1, padding=2, activation='gelu')
    y = layer(x, g)
    print(y.shape)

    x = torch.ones([10, 32, 32, 32, 4])
    g = torch.ones([10, 32, 32, 32, 3])
    layer = FNO3d(in_dim=4, out_dim=1, modes=(8, 8, 8), width=32, depth=4, steps=1, padding=6, activation='gelu')
    y = layer(x, g)
    print(y.shape)