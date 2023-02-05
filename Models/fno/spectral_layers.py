#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/6 17:37
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : spectral_layers.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_

from functools import partial
from Models.configs import activation_dict


class SpectralConv1d(nn.Module):
    '''
    1维谱卷积
    Modified Zongyi Li's Spectral1dConv code
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py
    '''

    def __init__(self, in_dim,
                 out_dim,
                 modes: int,  # number of fourier modes
                 dropout=0.1,
                 norm="ortho",
                 return_freq=False,
                 activation='gelu'):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.norm = norm
        self.dropout = dropout
        self.return_freq = return_freq
        self.activation = activation_dict[activation]
        self.linear = nn.Conv1d(self.in_dim, self.out_dim, 1)  # for residual

        self.scale = (1 / (in_dim * out_dim))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_dim, out_dim, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        """
        forward computation
        """
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, norm=self.norm)
        res = self.linear(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_dim, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        x = self.activation(x + res)

        if self.return_freq:
            return x, out_ft
        else:
            return x


class SpectralConv2d(nn.Module):
    '''
    2维谱卷积
    Modified Zongyi Li's SpectralConv2d PyTorch 1.6 code
    using only real weights
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
    '''

    def __init__(self, in_dim,
                 out_dim,
                 modes: tuple,  # number of fourier modes
                 dropout=0.1,
                 norm='ortho',
                 activation='gelu',
                 return_freq=False):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes1 = modes[0]  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes[1]

        self.norm = norm
        self.dropout = dropout
        self.activation = activation_dict[activation]
        self.return_freq = return_freq
        self.linear = nn.Conv2d(self.in_dim, self.out_dim, 1)  # for residual

        self.scale = (1 / (in_dim * out_dim))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_dim, out_dim, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_dim, out_dim, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        forward computation
        """
        batch_size = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, norm=self.norm)
        res = self.linear(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, self.out_dim, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        x = self.activation(x + res)

        if self.return_freq:
            return x, out_ft
        else:
            return x


class SpectralConv3d(nn.Module):
    '''
    三维谱卷积
    Modified Zongyi Li's SpectralConv2d PyTorch 1.6 code
    using only real weights
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
    '''

    def __init__(self, in_dim,
                 out_dim,
                 modes: tuple,
                 dropout=0.1,
                 norm='ortho',
                 activation='silu',
                 return_freq=False):  # whether to return the frequency target
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes1 = modes[0]  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes[1]
        self.modes3 = modes[2]

        self.norm = norm
        self.dropout = dropout
        self.return_freq = return_freq
        self.activation = activation_dict[activation]

        self.linear = nn.Conv3d(self.in_dim, self.out_dim, 1)  # for residual

        self.scale = (1 / (in_dim * out_dim))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_dim, out_dim, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_dim, out_dim, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_dim, out_dim, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_dim, out_dim, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        """
        forward computation
        """
        batch_size = x.size(0)
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm=self.norm)
        res = self.linear(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, self.out_dim, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        x = self.activation(x + res)

        if self.return_freq:
            return x, out_ft
        else:
            return x


if __name__ == '__main__':
    x = torch.ones([10, 3, 64])
    layer = SpectralConv1d(in_dim=3, out_dim=10, modes=5)
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 3, 55, 64])
    layer = SpectralConv2d(in_dim=3, out_dim=10, modes=(5, 3))
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 3, 16, 32, 48])
    layer = SpectralConv3d(in_dim=3, out_dim=4, modes=(5, 5, 5))
    y = layer(x)
    print(y.shape)
