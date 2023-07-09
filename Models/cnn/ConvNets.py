#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/26 3:16
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : ConvNets.py
"""
from cnn.conv_layers import *
from basic.basic_layers import *

class UpSampleNet1d(nn.Module):
    """
        1维上采样卷积网络
    """

    def __init__(self, in_sizes: int, out_sizes: tuple, width=32, depth=4,
                 activation='gelu', dropout=0.0):

        super(UpSampleNet1d, self).__init__()
        self.out_sizes = out_sizes[:-1]
        self.out_dim = out_sizes[-1]
        self.width = width
        self.depth = depth
        self.in_dim = in_sizes
        self.dropout = dropout

        self.depth = math.floor(math.log2(self.out_sizes[0])) - 1

        self.hidden_size = [0]
        self.hidden_size[0] = 2 ** math.floor(math.log2(self.out_sizes[0] / 2 ** self.depth))

        self.linear = nn.Linear(self.in_dim, math.prod(self.hidden_size) * self.width)

        self.upconvs = nn.ModuleList()
        for i in range(self.depth):
            self.upconvs.append(
                Interp1dUpsample(self.width, self.width, residual=True, conv_block=True,
                                activation=activation, dropout=self.dropout,
                                 interp_size=self.hidden_size, )
            )
        self.interp_out = Interp1dUpsample(in_dim=width, out_dim=self.out_dim, residual=False, conv_block=True,
                                           activation=activation, dropout=self.dropout,
                                           interp_size=self.out_sizes, )
        self.conv = nn.Conv1d(self.out_dim, self.out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        forward computation
        """
        x = self.linear(x)
        x = x.view([-1, self.width] + self.hidden_size)
        for i in range(self.depth):
            x = self.upconvs[i](x)
        x = self.interp_out(x)
        x = self.conv(x)
        return x.permute(0, 2, 1)


class DownSampleNet1d(nn.Module):
    """
        1维下采样卷积网络
    """

    def __init__(self, in_sizes: tuple, out_sizes: int, width=32, depth=4, activation='gelu', dropout=0.0):

        super(DownSampleNet1d, self).__init__()

        self.in_sizes = in_sizes[:-1]
        self.in_dim = in_sizes[-1]
        self.width = width
        self.depth = depth
        self.out_dim = out_sizes
        self.dropout = dropout
        self.activation = activation
        log2_in = [math.floor(math.log2(self.in_sizes[0]))]

        self.depth = min(log2_in[0] - 1, depth)
        self._out_size = [2 ** (log2_in[0] - self.depth)]

        self.interp_in = Interp1dUpsample(in_dim=self.in_dim, out_dim=self.width, residual=False, conv_block=True,
                                          activation=self.activation, dropout=self.dropout,
                                          interp_size=(2 ** log2_in[0]))
        self.downconvs = nn.ModuleList()
        for i in range(self.depth):
            self.downconvs.append(nn.Sequential(
                Conv1dResBlock(self.width, self.width, basic_block=True, activation=activation, dropout=dropout),
                nn.AvgPool1d(2, 2), ))

        self.linear = nn.Sequential(nn.Linear(math.prod(self._out_size) * self.width, 64),
                                    activation_dict[activation],
                                    nn.Linear(64, self.out_dim)
                                    )

    def forward(self, x):
        """
        forward computation
        """
        x = x.permute(0, 2, 1)
        x = self.interp_in(x)
        for i in range(self.depth):
            x = self.downconvs[i](x)
        x = x.reshape([-1, math.prod(self._out_size) * self.width])
        x = self.linear(x)
        return x


class UNet1d(nn.Module):
    """
        1维Unet  UNET model: https://github.com/milesial/Pytorch-UNet
    """

    def __init__(self, in_sizes: tuple, out_sizes: tuple, width=32, depth=4, steps=1, activation='gelu',
                 dropout=0.0):
        """
        :param in_sizes: (C_in, H_in)
        :param out_sizes: (C_out, H_out)
        :param width: hidden dim, int
        :param depth: hidden layers, int
        """
        super(UNet1d, self).__init__()

        self.in_sizes = in_sizes[:-1]
        self.out_sizes = out_sizes[:-1]
        self.in_dim = in_sizes[-1]
        self.out_dim = out_sizes[-1]
        self.width = width
        self.depth = depth
        self.steps = steps

        self._input_sizes = [0]
        self._input_sizes[0] = max(2 ** math.floor(math.log2(in_sizes[0])), 2 ** depth)
        # self._input_sizes[1] = max(2 ** math.floor(math.log2(in_sizes[1])), 2 ** depth)

        self.interp_in = Interp1dUpsample(in_dim=steps*self.in_dim + 1, out_dim=self.in_dim, activation=activation,
                                          dropout=dropout,
                                          interp_size=self._input_sizes, conv_block=True)
        self.encoders = nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
                self.encoders.append(
                    Conv1dResBlock(self.in_dim, width, basic_block=True, activation=activation, dropout=dropout))
            else:
                self.encoders.append(nn.Sequential(nn.MaxPool1d(2, 2),
                                                   Conv1dResBlock(2 ** (i - 1) * width, 2 ** i * width,
                                                                  basic_block=True, activation=activation,
                                                                  dropout=dropout)))
        self.bottleneck = nn.Sequential(nn.MaxPool1d(2, 2),
                                        Conv1dResBlock(2 ** i * width, 2 ** i * width * 2, basic_block=True,
                                                       activation=activation, dropout=dropout))

        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        for i in range(self.depth, 0, -1):
            self.decoders.append(
                Conv1dResBlock(2 ** i * width, 2 ** (i - 1) * width, basic_block=True, activation=activation,
                               dropout=dropout))
            self.upconvs.append(
                DeConv1dBlock(2 ** i * width, 2 ** (i - 1) * width, 2 ** (i - 1) * width, activation=activation,
                              dropout=dropout))

        self.conv1 = Conv1dResBlock(in_dim=width, out_dim=self.out_dim, basic_block=False, activation=activation,
                                    dropout=dropout)

        self.interp_out = Interp1dUpsample(in_dim=self.out_dim, out_dim=self.out_dim, interp_size=self.out_sizes,
                                           conv_block=False, activation=activation, dropout=dropout)

        self.conv2 = nn.Conv1d(self.out_dim, self.out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, grid):
        """
        forward computation
        """
        # x dim = [b, x1, t*v]
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 2, 1)

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
        return x.permute(0, 2, 1)


class UpSampleNet2d(nn.Module):
    """
        2维上采样卷积网络
    """

    def __init__(self, in_sizes: int, out_sizes: tuple, width=32, depth=4,
                 activation='gelu', dropout=0.0):
        """
        :param in_sizes: C_in, int
        :param out_sizes: (H_out, W_out, C_out)
        :param width: hidden dim, int
        :param depth: hidden layers, int maybe adjust based on the in_sizes
        :param activation: str 'gelu' 'relu' 'silu' 'tanh'
        :param dropout: dropout, float
        """
        super(UpSampleNet2d, self).__init__()
        self.out_sizes = out_sizes[:-1]
        self.out_dim = out_sizes[-1]
        self.width = width
        self.depth = depth
        self.in_dim = in_sizes
        self.dropout = dropout
        self.depth = min(math.floor(math.log2(self.out_sizes[0])) - 1, math.floor(math.log2(self.out_sizes[1])) - 1,
                         depth)
        self.hidden_size = [0, 0]
        self.hidden_size[0] = 2 ** math.floor(math.log2(self.out_sizes[0] / 2 ** self.depth))
        self.hidden_size[1] = 2 ** math.floor(math.log2(self.out_sizes[1] / 2 ** self.depth))

        self.linear = nn.Linear(self.in_dim, math.prod(self.hidden_size) * self.width)

        self.upconvs = nn.ModuleList()
        for i in range(self.depth):
            self.upconvs.append(
                Interp2dUpsample(self.width, self.width, residual=True, conv_block=True,
                                 interp_mode='bilinear', activation=activation, dropout=self.dropout,
                                 interp_size=(self.hidden_size[0] * 2 ** (i + 1), self.hidden_size[1] * 2 ** (i + 1)), )
            )
        self.interp_out = Interp2dUpsample(in_dim=width, out_dim=self.out_dim, residual=False, conv_block=True,
                                           interp_mode='bilinear', activation=activation, dropout=self.dropout,
                                           interp_size=self.out_sizes, )
        self.conv = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, x):
        """
        forward computation
        """
        x = self.linear(x)
        x = x.view([-1, self.width] + self.hidden_size)
        for i in range(self.depth):
            x = self.upconvs[i](x)
        x = self.interp_out(x)
        x = self.conv(x)
        return x.permute(0, 2, 3, 1)


class DownSampleNet2d(nn.Module):
    """
        2维下采样卷积网络
    """

    def __init__(self, in_sizes: tuple, out_sizes: int, width=32, depth=4, activation='gelu', dropout=0.0):
        """
        :param in_sizes: (H_in, W_in, C_in)
        :param out_sizes: C_out, int
        :param width: hidden dim, int
        :param depth: hidden layers, int maybe adjust based on the in_sizes
        :param activation: str 'gelu' 'relu' 'silu' 'tanh'
        :param dropout: dropout, float
        """
        super(DownSampleNet2d, self).__init__()

        self.in_sizes = in_sizes[:-1]
        self.in_dim = in_sizes[-1]
        self.width = width
        self.depth = depth
        self.out_dim = out_sizes
        self.dropout = dropout
        self.activation = activation
        log2_in = [math.floor(math.log2(self.in_sizes[0])), math.floor(math.log2(self.in_sizes[1]))]
        self.depth = min(log2_in[0] - 1, log2_in[1] - 1, depth)
        self._out_size = [2 ** (log2_in[0] - self.depth), 2 ** (log2_in[1] - self.depth)]

        self.interp_in = Interp2dUpsample(in_dim=self.in_dim, out_dim=self.width, residual=False, conv_block=True,
                                          interp_mode='bilinear', activation=self.activation, dropout=self.dropout,
                                          interp_size=(2 ** log2_in[0], 2 ** log2_in[1]))
        self.downconvs = nn.ModuleList()
        for i in range(self.depth):
            self.downconvs.append(nn.Sequential(
                Conv2dResBlock(self.width, self.width, basic_block=True, activation=activation, dropout=dropout),
                nn.AvgPool2d(2, 2), ))

        self.linear = nn.Sequential(nn.Linear(math.prod(self._out_size) * self.width,
                                              int(math.prod(self._out_size) * self.width / 4)),
                                    activation_dict[activation],
                                    nn.Linear(int(math.prod(self._out_size) * self.width / 4), self.out_dim)
                                    )

    def forward(self, x):
        """
        forward computation
        """
        x = x.permute(0, 3, 1, 2)
        x = self.interp_in(x)
        for i in range(self.depth):
            x = self.downconvs[i](x)
        x = x.reshape([-1, math.prod(self._out_size) * self.width])
        x = self.linear(x)
        return x


class UNet2d(nn.Module):
    """
        2维UNet
    """

    def __init__(self, in_sizes: tuple, out_sizes: tuple, width=32, depth=4, steps=1, activation='gelu',
                 dropout=0.0):
        """
        :param in_sizes: (H_in, W_in, C_in)
        :param out_sizes: (H_out, W_out, C_out)
        :param width: hidden dim, int
        :param depth: hidden layers, int
        """
        super(UNet2d, self).__init__()

        self.in_sizes = in_sizes[:-1]
        self.out_sizes = out_sizes[:-1]
        self.in_dim = in_sizes[-1]
        self.out_dim = out_sizes[-1]
        self.width = width
        self.depth = depth
        self.steps = steps

        self._input_sizes = [0, 0]
        self._input_sizes[0] = max(2 ** math.floor(math.log2(self.in_sizes[0])), 2 ** depth)
        self._input_sizes[1] = max(2 ** math.floor(math.log2(self.in_sizes[1])), 2 ** depth)


        self.interp_in = Interp2dUpsample(in_dim=steps*self.in_dim + 2, out_dim=self.in_dim, activation=activation,
                                          dropout=dropout, interp_size=self._input_sizes, conv_block=True)
        self.encoders = nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
                self.encoders.append(
                    Conv2dResBlock(self.in_dim, width, basic_block=True, activation=activation, dropout=dropout))
            else:
                self.encoders.append(nn.Sequential(nn.MaxPool2d(2),
                                                   Conv2dResBlock(2 ** (i - 1) * width, 2 ** i * width,
                                                                  basic_block=True, activation=activation,
                                                                  dropout=dropout)))

        self.bottleneck = nn.Sequential(nn.MaxPool2d(2),
                                        Conv2dResBlock(2 ** i * width, 2 ** i * width * 2, basic_block=True,
                                                       activation=activation, dropout=dropout))

        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        for i in range(self.depth, 0, -1):
            self.decoders.append(
                Conv2dResBlock(2 ** i * width, 2 ** (i - 1) * width, activation=activation,
                               basic_block=True, dropout=dropout))
            self.upconvs.append(
                DeConv2dBlock(2 ** i * width, 2 ** (i - 1) * width, 2 ** (i - 1) * width, activation=activation,
                              dropout=dropout))

        self.conv1 = Conv2dResBlock(in_dim=width, out_dim=self.out_dim, basic_block=False, activation=activation,
                                    dropout=dropout)

        self.interp_out = Interp2dUpsample(in_dim=self.out_dim, out_dim=self.out_dim, interp_size=self.out_sizes,
                                           conv_block=False, activation=activation, dropout=dropout)

        self.conv2 = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, grid):
        """
        forward computation
        """
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2)
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
        return x.permute(0, 2, 3, 1)


class UpSampleNet3d(nn.Module):
    """
        3维上采样卷积网络
    """

    def __init__(self, in_sizes: int, out_sizes: tuple, width=32, depth=4,
                 activation='gelu', dropout=0.0):
        """
        :param in_sizes: C_in, int
        :param out_sizes: (H_out, W_out, C_out)
        :param width: hidden dim, int
        :param depth: hidden layers, int maybe adjust based on the in_sizes
        :param activation: str 'gelu' 'relu' 'silu' 'tanh'
        :param dropout: dropout, float
        """
        super(UpSampleNet3d, self).__init__()
        self.out_sizes = out_sizes[:-1]
        self.out_dim = out_sizes[-1]
        self.width = width
        self.depth = depth
        self.in_dim = in_sizes
        self.dropout = dropout
        self.depth = min(math.floor(math.log2(self.out_sizes[0])) - 1,
                         math.floor(math.log2(self.out_sizes[1])) - 1, math.floor(math.log2(self.out_sizes[2])) - 1,
                         depth)

        self.hidden_size = [0, 0, 0]
        self.hidden_size[0] = 2 ** math.floor(math.log2(self.out_sizes[0] / 2 ** self.depth))
        self.hidden_size[1] = 2 ** math.floor(math.log2(self.out_sizes[1] / 2 ** self.depth))
        self.hidden_size[2] = 2 ** math.floor(math.log2(self.out_sizes[2] / 2 ** self.depth))

        self.linear = nn.Linear(self.in_dim, math.prod(self.hidden_size) * self.width)

        self.upconvs = nn.ModuleList()
        for i in range(self.depth):
            self.upconvs.append(
                Interp3dUpsample(self.width, self.width, residual=True, conv_block=True,
                                 activation=activation, dropout=self.dropout,
                                 interp_size=(self.hidden_size[0] * 2 ** (i + 1), self.hidden_size[1] * 2 ** (i + 1), self.hidden_size[2] * 2 ** (i + 1)), )
            )
        self.interp_out = Interp3dUpsample(in_dim=width, out_dim=self.out_dim, residual=False, conv_block=True,
                                           activation=activation, dropout=self.dropout,
                                           interp_size=self.out_sizes, )
        self.conv = nn.Conv3d(self.out_dim, self.out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        forward computation
        """
        x = self.linear(x)
        x = x.view([-1, self.width] + self.hidden_size)
        for i in range(self.depth):
            x = self.upconvs[i](x)
        x = self.interp_out(x)
        x = self.conv(x)
        return x.permute(0, 2, 3, 4, 1)


class DownSampleNet3d(nn.Module):
    """
        3维下采样卷积网络
    """

    def __init__(self, in_sizes: tuple, out_sizes: int, width=32, depth=4, activation='gelu', dropout=0.0):
        """
        :param in_sizes: (H_in, W_in, C_in)
        :param out_sizes: C_out, int
        :param width: hidden dim, int
        :param depth: hidden layers, int maybe adjust based on the in_sizes
        :param activation: str 'gelu' 'relu' 'silu' 'tanh'
        :param dropout: dropout, float
        """
        super(DownSampleNet3d, self).__init__()

        self.in_sizes = in_sizes[:-1]
        self.in_dim = in_sizes[-1]
        self.width = width
        self.depth = depth
        self.out_dim = out_sizes
        self.dropout = dropout
        self.activation = activation
        log2_in = [math.floor(math.log2(self.in_sizes[0])), math.floor(math.log2(self.in_sizes[1])),  math.floor(math.log2(self.in_sizes[2]))]
        self.depth = min(log2_in[0] - 1, log2_in[1] - 1,log2_in[2] - 1, depth)
        self._out_size = [2 ** (log2_in[0] - self.depth), 2 ** (log2_in[1] - self.depth),  2 ** (log2_in[2] - self.depth)]

        self.interp_in = Interp3dUpsample(in_dim=self.in_dim, out_dim=self.width, residual=False, conv_block=True,
                                          activation=self.activation, dropout=self.dropout,
                                          interp_size=(2 ** log2_in[0], 2 ** log2_in[1], 2 ** log2_in[2]))
        self.downconvs = nn.ModuleList()
        for i in range(self.depth):
            self.downconvs.append(nn.Sequential(
                Conv3dResBlock(self.width, self.width, basic_block=True, activation=activation, dropout=dropout),
                nn.AvgPool3d(2, 2), ))

        self.linear = nn.Sequential(nn.Linear(math.prod(self._out_size) * self.width, 64),
                                    activation_dict[activation],
                                    nn.Linear(64, self.out_dim)
                                    )

    def forward(self, x):
        """
        forward computation
        """
        x = x.permute(0, 4, 1, 2, 3)
        x = self.interp_in(x)
        for i in range(self.depth):
            x = self.downconvs[i](x)
        x = x.reshape([-1, math.prod(self._out_size) * self.width])
        x = self.linear(x)
        return x


class UNet3d(nn.Module):
    """
        3维UNet
    """

    def __init__(self, in_sizes: tuple, out_sizes: tuple, width=32, depth=4, steps=1, activation='gelu',
                 dropout=0.0):

        super(UNet3d, self).__init__()

        self.in_sizes = in_sizes[:-1]
        self.out_sizes = out_sizes[:-1]
        self.in_dim = in_sizes[-1]
        self.out_dim = out_sizes[-1]
        self.width = width
        self.depth = depth
        self.steps = steps

        self._input_sizes = [0, 0, 0]
        self._input_sizes[0] = max(2 ** math.floor(math.log2(in_sizes[0])), 2 ** depth)
        self._input_sizes[1] = max(2 ** math.floor(math.log2(in_sizes[1])), 2 ** depth)
        self._input_sizes[2] = max(2 ** math.floor(math.log2(in_sizes[2])), 2 ** depth)

        self.interp_in = Interp3dUpsample(in_dim=steps*self.in_dim + 3, out_dim=self.in_dim, activation=activation,
                                          dropout=dropout,
                                          interp_size=self._input_sizes, conv_block=True)
        self.encoders = nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
                self.encoders.append(
                    Conv3dResBlock(self.in_dim, width, basic_block=True, activation=activation, dropout=dropout))
            else:
                self.encoders.append(nn.Sequential(nn.MaxPool3d(2, 2),
                                                   Conv3dResBlock(2 ** (i - 1) * width, 2 ** i * width,
                                                                  basic_block=True, activation=activation,
                                                                  dropout=dropout)))

        self.bottleneck = nn.Sequential(nn.MaxPool3d(2, 2),
                                        Conv3dResBlock(2 ** i * width, 2 ** i * width * 2, basic_block=True,
                                                       activation=activation, dropout=dropout))

        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        for i in range(self.depth, 0, -1):
            self.decoders.append(
                Conv3dResBlock(2 ** i * width, 2 ** (i - 1) * width, activation=activation,
                               basic_block=True, dropout=dropout))
            self.upconvs.append(
                DeConv3dBlock(2 ** i * width, 2 ** (i - 1) * width, 2 ** (i - 1) * width, activation=activation,
                              dropout=dropout))

        self.conv1 = Conv3dResBlock(in_dim=width, out_dim=self.out_dim, basic_block=False, activation=activation,
                                    dropout=dropout)

        self.interp_out = Interp3dUpsample(in_dim=self.out_dim, out_dim=self.out_dim, interp_size=self.out_sizes,
                                           conv_block=False, activation=activation, dropout=dropout)

        self.conv2 = nn.Conv3d(self.out_dim, self.out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, grid):
        """
        forward computation
        """
        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 4, 1, 2, 3)
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
        return x.permute(0, 2, 3, 4, 1)

if __name__ == '__main__':

    x = torch.ones([10, 92, 8])
    g = torch.ones([10, 92, 1])
    input_size = x.shape[1:]
    layer = UNet1d(in_sizes=input_size, out_sizes=(128, 16), width=32, depth=6, steps=1)
    y = layer(x, g)
    print(y.shape)

    x = torch.ones([10, 10])
    in_sizes, out_sizes = 10, (58, 32)
    layer = UpSampleNet1d(in_sizes, out_sizes, width=32, depth=4)
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 92, 8])
    in_sizes, out_sizes = x.shape[1:], 10
    layer = DownSampleNet1d(in_sizes, out_sizes, width=32, depth=4)
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 4, 92, 8])
    g = torch.ones([10, 4, 92, 2])
    input_size = x.shape[1:]
    layer = UNet2d(in_sizes=input_size, out_sizes=(32, 32, 5), width=32, depth=6, steps=1)
    y = layer(x, g)
    print(y.shape)

    x = torch.ones([10, 10])
    in_sizes, out_sizes = 10, (58, 32, 5)
    layer = UpSampleNet2d(in_sizes, out_sizes, width=32, depth=4)
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 22, 92, 4])
    in_sizes, out_sizes = x.shape[1:], 10
    layer = DownSampleNet2d(in_sizes, out_sizes, width=32, depth=4)
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 32, 92, 92, 8])
    g = torch.ones([10, 32, 92, 92, 3])
    input_size = x.shape[1:]
    layer = UNet3d(in_sizes=input_size, out_sizes=(32, 32, 16, 5), width=32, depth=6)
    y = layer(x, g)
    print(y.shape)

    x = torch.ones([10, 10])
    in_sizes, out_sizes = 10, (32, 58, 32, 8)
    layer = UpSampleNet3d(in_sizes, out_sizes, width=32, depth=4)
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 4, 92, 52, 8])
    in_sizes, out_sizes = x.shape[1:], 5
    layer = DownSampleNet3d(in_sizes, out_sizes, width=32, depth=4)
    y = layer(x)
    print(y.shape)