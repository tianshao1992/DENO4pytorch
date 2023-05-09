#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/6 17:37
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : spectral_layers.py
"""
import math
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
        self.dropout = nn.Dropout(dropout)
        self.return_freq = return_freq
        self.activation = activation_dict[activation]
        self.linear = nn.Conv1d(self.in_dim, self.out_dim, 1)  # for residual
        # self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.scale = (1 / (in_dim * out_dim))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_dim, out_dim, self.modes, dtype=torch.cfloat))
        # xavier_normal_(self.weights1, gain=1 / (in_dim * out_dim))

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
        res = self.linear(x)
        # x = self.dropout(x)
        x_ft = torch.fft.rfft(x, norm=self.norm)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_dim, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, norm=self.norm)
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
        if isinstance(modes, int):
            self.modes1 = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes
        else:
            self.modes1 = modes[0]  # Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes[1]

        self.norm = norm
        self.dropout = nn.Dropout(dropout)
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
        res = self.linear(x)
        x = self.dropout(x)
        x_ft = torch.fft.rfft2(x, norm=self.norm)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, self.out_dim, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm=self.norm)
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
        if isinstance(modes, int):
            self.modes1 = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes
            self.modes3 = modes
        else:
            self.modes1 = modes[0]  # Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes[1]
            self.modes3 = modes[2]

        self.norm = norm
        self.dropout = nn.Dropout(dropout)
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
        res = self.linear(x)
        # x = self.dropout(x)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm=self.norm)
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
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)), norm=self.norm)
        x = self.activation(x + res)

        if self.return_freq:
            return x, out_ft
        else:
            return x


class AdaptiveFourier1d(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """

    def __init__(self, hidden_size, num_blocks=8,
                 sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1, activation='relu'):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.activation = activation_dict[activation]

        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor,
                                    dtype=torch.cfloat))
        self.bias1 = nn.Parameter(
            self.scale * torch.rand(self.num_blocks, self.block_size * self.hidden_size_factor,
                                    dtype=torch.cfloat))
        self.bias2 = nn.Parameter(
            self.scale * torch.rand(self.num_blocks, self.block_size,
                                    dtype=torch.cfloat))

    def compl_mul1d(self, input, weights, bias):
        # (batch, num_blocks, block_size, l), (num_blocks, block_size, block_size * hidden_size_factor)
        # -> (batch, num_blocks, block_size * hidden_size_factor, l)
        return torch.einsum("...il, ...io->...ol", input, weights) + bias[..., None]

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, N = x.shape

        x = torch.fft.rfft(x, norm="ortho")
        x = x.reshape(B, self.num_blocks, self.block_size, N // 2 + 1)

        out_ft1 = torch.zeros([B, self.num_blocks, self.block_size * self.hidden_size_factor, N // 2 + 1],
                              dtype=torch.cfloat, device=x.device)
        out_ft2 = torch.zeros(x.shape, dtype=torch.cfloat, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        # note that the activation function doesn't support complex type
        out_ft1[..., :kept_modes] = torch.view_as_complex(
            self.activation(torch.view_as_real(self.compl_mul1d(x[..., :kept_modes], self.weights1, self.bias1))))

        out_ft2[..., :kept_modes] = self.compl_mul1d(out_ft1[..., :kept_modes], self.weights2, self.bias2)

        x = torch.view_as_complex(
            F.softshrink(torch.view_as_real(out_ft2), lambd=self.sparsity_threshold))

        x = x.reshape(B, C, N // 2 + 1)
        x = torch.fft.irfft(x, n=N, norm="ortho")
        x = x.type(dtype)
        return x + bias


class AdaptiveFourier2d(nn.Module):
    """
    Modified Zongyi Li's AFNO2d PyTorch 1.6 code
    using only real weights
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py

    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """

    def __init__(self, hidden_size, num_blocks=8,
                 sparsity_threshold=0.01,
                 hard_thresholding_fraction=1,
                 hidden_size_factor=1, activation='gelu'):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.activation = activation_dict[activation]

        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor,
                                    dtype=torch.cfloat))
        self.bias1 = nn.Parameter(
            self.scale * torch.rand(self.num_blocks, self.block_size * self.hidden_size_factor,
                                    dtype=torch.cfloat))
        self.bias2 = nn.Parameter(
            self.scale * torch.rand(self.num_blocks, self.block_size,
                                    dtype=torch.cfloat))

    def compl_mul2d(self, input, weights, bias):
        # (batch, num_blocks, block_size, H, W), (num_blocks, block_size, block_size * hidden_size_factor)
        # -> (batch, num_blocks, block_size * hidden_size_factor, H, W)
        return torch.einsum("...ijk, ...io->...ojk", input, weights) + bias[..., None, None]

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape

        # x = x.reshape(B, C, H, W, D)
        x = torch.fft.rfft2(x, norm="ortho")

        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[-2], x.shape[-1])

        out_ft1 = torch.zeros([B, self.num_blocks, self.block_size * self.hidden_size_factor,
                               x.shape[-2], x.shape[-1]], dtype=torch.cfloat, device=x.device)
        out_ft2 = torch.zeros(x.shape, dtype=torch.cfloat, device=x.device)

        total_modes = (H * W) // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        # note that the activation function doesn't support complex type
        out_ft1[..., :kept_modes] = torch.view_as_complex(
            self.activation(torch.view_as_real(self.compl_mul2d(x[..., :kept_modes], self.weights1, self.bias1))))

        out_ft2[..., :kept_modes] = self.compl_mul2d(out_ft1[..., :kept_modes], self.weights2, self.bias2)

        x = torch.view_as_complex(
            F.softshrink(torch.view_as_real(out_ft2), lambd=self.sparsity_threshold))

        x = x.reshape(B, C, x.shape[-2], x.shape[-1])
        x = torch.fft.irfft2(x, s=(H, W), norm="ortho")
        x = x.type(dtype)

        return x + bias


class AdaptiveFourier3d(nn.Module):
    """
    Modified Zongyi Li's AFNO2d PyTorch 1.6 code
    using only real weights
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py

    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """

    def __init__(self, hidden_size, num_blocks=8,
                 sparsity_threshold=0.01,
                 hard_thresholding_fraction=1,
                 hidden_size_factor=1, activation='gelu'):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.activation = activation_dict[activation]

        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor,
                                    dtype=torch.cfloat))
        self.bias1 = nn.Parameter(
            self.scale * torch.rand(self.num_blocks, self.block_size * self.hidden_size_factor,
                                    dtype=torch.cfloat))
        self.bias2 = nn.Parameter(
            self.scale * torch.rand(self.num_blocks, self.block_size,
                                    dtype=torch.cfloat))

    def compl_mul3d(self, input, weights, bias):
        # (batch, num_blocks, block_size, H, W), (num_blocks, block_size, block_size * hidden_size_factor)
        # -> (batch, num_blocks, block_size * hidden_size_factor, H, W)
        return torch.einsum("...ijkl, ...io->...ojkl", input, weights) + bias[..., None, None, None]

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W, D = x.shape

        # x = x.reshape(B, C, H, W, D)
        x = torch.fft.rfftn(x, dim=[-3, -2, -1], norm="ortho")

        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[-3], x.shape[-2], x.shape[-1])

        out_ft1 = torch.zeros([B, self.num_blocks, self.block_size * self.hidden_size_factor,
                               x.shape[-3], x.shape[-2], x.shape[-1]], dtype=torch.cfloat, device=x.device)

        out_ft2 = torch.zeros(x.shape, dtype=torch.cfloat, device=x.device)

        total_modes = (H * W * D) // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        # note that the activation function doesn't support complex type
        out_ft1[..., :kept_modes] = torch.view_as_complex(
            self.activation(torch.view_as_real(self.compl_mul3d(x[..., :kept_modes], self.weights1, self.bias1))))

        out_ft2[..., :kept_modes] = self.compl_mul3d(out_ft1[..., :kept_modes], self.weights2, self.bias2)

        x = torch.view_as_complex(
            F.softshrink(torch.view_as_real(out_ft2), lambd=self.sparsity_threshold))

        x = x.reshape(B, C, x.shape[-3], x.shape[-2], x.shape[-1])
        x = torch.fft.irfftn(x, s=(H, W, D), norm="ortho")
        x = x.type(dtype)

        return x + bias


if __name__ == '__main__':
    x = torch.ones([10, 3, 64])
    layer = SpectralConv1d(in_dim=3, out_dim=10, modes=5)
    y = layer(x)
    print(y.shape)

    lossfunc = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)
    loss = lossfunc(y, torch.ones_like(y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    x = torch.ones([10, 3, 55, 64])
    layer = SpectralConv2d(in_dim=3, out_dim=10, modes=(5, 3))
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 3, 16, 32, 48])
    layer = SpectralConv3d(in_dim=3, out_dim=4, modes=(5, 5, 5))
    y = layer(x)
    print(y.shape)

    x = torch.ones(10, 64, 128)
    layer = AdaptiveFourier1d(hidden_size=64, num_blocks=4)
    y = layer(x)
    print(y.shape)

    x = torch.ones(10, 64, 55, 64)
    layer = AdaptiveFourier2d(hidden_size=64, num_blocks=4)
    y = layer(x)
    print(y.shape)

    x = torch.ones(10, 64, 55, 64, 33)
    layer = AdaptiveFourier3d(hidden_size=64, num_blocks=4)
    y = layer(x)
    print(y.shape)
