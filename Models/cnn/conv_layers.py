#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/25 23:33
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : conv_layers.py
"""
from basic.basic_layers import Identity
import torch.nn.functional as F
from Models.configs import *


class Conv1dResBlock(nn.Module):
    """
        1D残差卷积块
    """

    def __init__(self, in_dim, out_dim,
                 kernel_size=3,
                 padding=1,
                 dilation=1,
                 dropout=0.1,
                 stride=1,
                 bias=False,
                 residual=False,
                 activation='silu',
                 basic_block=False,
                 ):
        super(Conv1dResBlock, self).__init__()

        self.activation = activation_dict[activation]
        self.add_res = residual
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, out_dim,
                      kernel_size=kernel_size,
                      padding=padding,
                      dilation=dilation,
                      stride=stride,
                      bias=bias),
            nn.Dropout(dropout),
        )
        self.basic_block = basic_block
        if self.basic_block:
            self.conv1 = nn.Sequential(
                nn.BatchNorm1d(num_features=out_dim),
                self.activation,
                nn.Conv1d(out_dim, out_dim,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=bias),
                nn.Dropout(dropout),
                nn.BatchNorm1d(num_features=out_dim),
            )
        self.apply_shortcut = (in_dim != out_dim)

        if self.add_res:
            if self.apply_shortcut:
                self.res = Identity(in_dim, out_dim)
            else:
                self.res = Identity()

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch_size, input_dim, H)
        """
        if self.add_res:
            h = self.res(x)

        x = self.conv(x)

        if self.basic_block:
            x = self.conv1(x)

        if self.add_res:
            return self.activation(x + h)
        else:
            return self.activation(x)


class DeConv1dBlock(nn.Module):
    """
        1D反卷积块
    """

    def __init__(self, in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 stride: int = 2,
                 kernel_size: int = 2,
                 # padding: int = 2,
                 # output_padding: int = 1,
                 dropout=0.1,
                 activation='silu'):
        super(DeConv1dBlock, self).__init__()
        # assert stride*2 == scaling_factor
        # padding1 = padding // 2 if padding // 2 >= 1 else 1

        self.deconv0 = nn.ConvTranspose1d(in_channels=in_dim,
                                          out_channels=hidden_dim,
                                          kernel_size=kernel_size,
                                          stride=stride
                                          # output_padding=output_padding,
                                          # padding=padding
                                          )
        self.conv0 = nn.Conv1d(in_channels=hidden_dim,
                               out_channels=out_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        # self.deconv1 = nn.ConvTranspose2d(in_channels=hidden_dim,
        #                                   out_channels=out_dim,
        #                                   kernel_size=kernel_size,
        #                                   stride=stride,
        #                                   # output_padding=output_padding,
        #                                   # padding=padding1,  # hard code bad, 1: for 85x85 grid, 2 for 43x43 grid
        #                                   )
        self.activation = activation_dict[activation]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch_size, input_dim, H)
        """
        x = self.deconv0(x)
        x = self.dropout(x)
        # x = self.activation(x)
        # x = self.conv0(x)
        # x = self.deconv1(x)
        # x = self.activation(x)
        return x


class Interp1dUpsample(nn.Module):
    """
        1维上采样插值块
    """

    def __init__(self, in_dim: int,
                 out_dim: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 residual=False,
                 conv_block=True,
                 interp_mode='linear',
                 interp_size=None,
                 activation='silu',
                 dropout=0.1):
        super(Interp1dUpsample, self).__init__()
        self.activation = activation_dict[activation]
        self.dropout = nn.Dropout(dropout)
        if conv_block:
            self.conv = nn.Sequential(Conv1dResBlock(
                in_dim, out_dim,
                kernel_size=kernel_size,
                padding=padding,
                residual=residual,
                dropout=dropout,
                activation=activation),
                self.dropout,
                self.activation)
        self.conv_block = conv_block
        self.interp_size = interp_size
        self.interp_mode = interp_mode

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch_size, input_dim, H)
        """
        if self.conv_block:
            x = self.conv(x)

        x = F.interpolate(x, size=self.interp_size,
                          mode=self.interp_mode,
                          align_corners=True)
        return x


class Conv2dResBlock(nn.Module):
    '''
    Conv2d + a residual block
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    Modified from ResNet's basic block, one conv less, no batchnorm
    No batchnorm
    '''

    def __init__(self, in_dim, out_dim,
                 kernel_size=3,
                 padding=1,
                 dilation=1,
                 dropout=0.1,
                 stride=1,
                 bias=False,
                 residual=False,
                 activation='silu',
                 basic_block=False,
                 ):
        super(Conv2dResBlock, self).__init__()

        self.activation = activation_dict[activation]
        self.add_res = residual
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim,
                      kernel_size=kernel_size,
                      padding=padding,
                      dilation=dilation,
                      stride=stride,
                      bias=bias),
            nn.Dropout(dropout),
        )
        self.basic_block = basic_block
        if self.basic_block:
            self.conv1 = nn.Sequential(
                self.activation,
                nn.Conv2d(out_dim, out_dim,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=bias),
                nn.Dropout(dropout),
            )
        self.apply_shortcut = (in_dim != out_dim)

        if self.add_res:
            if self.apply_shortcut:
                self.res = Identity(in_dim, out_dim)
            else:
                self.res = Identity()

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch_size, input_dim, H, W)
        """
        if self.add_res:
            h = self.res(x)

        x = self.conv(x)

        if self.basic_block:
            x = self.conv1(x)

        if self.add_res:
            return self.activation(x + h)
        else:
            return self.activation(x)


class DeConv2dBlock(nn.Module):
    '''
    Similar to a LeNet block
    4x upsampling, dimension hard-coded
    '''

    def __init__(self, in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 stride: int = 2,
                 kernel_size: int = 2,
                 # padding: int = 2,
                 # output_padding: int = 1,
                 dropout=0.1,
                 activation='silu'):
        super(DeConv2dBlock, self).__init__()
        # assert stride*2 == scaling_factor
        # padding1 = padding // 2 if padding // 2 >= 1 else 1

        self.deconv0 = nn.ConvTranspose2d(in_channels=in_dim,
                                          out_channels=hidden_dim,
                                          kernel_size=kernel_size,
                                          stride=stride
                                          # output_padding=output_padding,
                                          # padding=padding
                                          )
        self.conv0 = nn.Conv2d(in_channels=hidden_dim,
                               out_channels=out_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        # self.deconv1 = nn.ConvTranspose2d(in_channels=hidden_dim,
        #                                   out_channels=out_dim,
        #                                   kernel_size=kernel_size,
        #                                   stride=stride,
        #                                   # output_padding=output_padding,
        #                                   # padding=padding1,  # hard code bad, 1: for 85x85 grid, 2 for 43x43 grid
        #                                   )
        self.activation = activation_dict[activation]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch_size, input_dim, H, W)
        """
        x = self.deconv0(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.conv0(x)
        # x = self.deconv1(x)
        # x = self.activation(x)
        return x


class Interp2dUpsample(nn.Module):
    '''
    interpolate then Conv2dResBlock
    old code uses lambda and cannot be pickled
    temp hard-coded dimensions
    '''

    def __init__(self, in_dim: int,
                 out_dim: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 residual=False,
                 conv_block=True,
                 interp_mode='bilinear',
                 interp_size=None,
                 activation='silu',
                 dropout=0.1):
        super(Interp2dUpsample, self).__init__()
        self.activation = activation_dict[activation]
        self.dropout = nn.Dropout(dropout)
        if conv_block:
            self.conv = nn.Sequential(Conv2dResBlock(
                in_dim, out_dim,
                kernel_size=kernel_size,
                padding=padding,
                residual=residual,
                dropout=dropout,
                activation=activation),
                self.dropout,
                self.activation)
        self.conv_block = conv_block
        self.interp_size = interp_size
        self.interp_mode = interp_mode

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch_size, input_dim, H, W)
        """
        if self.conv_block:
            x = self.conv(x)

        x = F.interpolate(x, size=self.interp_size,
                          mode=self.interp_mode,
                          align_corners=True)
        return x


class Conv3dResBlock(nn.Module):
    """
        3维残差卷积块
    """

    def __init__(self, in_dim, out_dim,
                 kernel_size=3,
                 padding=1,
                 dilation=1,
                 dropout=0.1,
                 stride=1,
                 bias=False,
                 residual=False,
                 activation='silu',
                 basic_block=False,
                 ):
        super(Conv3dResBlock, self).__init__()

        self.activation = activation_dict[activation]
        self.add_res = residual
        self.conv = nn.Sequential(
            nn.Conv3d(in_dim, out_dim,
                      kernel_size=kernel_size,
                      padding=padding,
                      dilation=dilation,
                      stride=stride,
                      bias=bias),
            nn.Dropout(dropout),
        )
        self.basic_block = basic_block
        if self.basic_block:
            self.conv1 = nn.Sequential(
                self.activation,
                nn.Conv3d(out_dim, out_dim,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=bias),
                nn.Dropout(dropout),
            )
        self.apply_shortcut = (in_dim != out_dim)

        if self.add_res:
            if self.apply_shortcut:
                self.res = Identity(in_dim, out_dim)
            else:
                self.res = Identity()

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch_size, input_dim, H, W, L)
        """
        if self.add_res:
            h = self.res(x)

        x = self.conv(x)

        if self.basic_block:
            x = self.conv1(x)

        if self.add_res:
            return self.activation(x + h)
        else:
            return self.activation(x)


class DeConv3dBlock(nn.Module):
    """
        3维反卷积块
    """

    def __init__(self, in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 stride: int = 2,
                 kernel_size: int = 2,
                 # padding: int = 2,
                 # output_padding: int = 1,
                 dropout=0.1,
                 activation='silu'):
        super(DeConv3dBlock, self).__init__()
        # assert stride*2 == scaling_factor
        # padding1 = padding // 2 if padding // 2 >= 1 else 1

        self.deconv0 = nn.ConvTranspose3d(in_channels=in_dim,
                                          out_channels=hidden_dim,
                                          kernel_size=kernel_size,
                                          stride=stride
                                          # output_padding=output_padding,
                                          # padding=padding
                                          )
        self.conv0 = nn.Conv3d(in_channels=hidden_dim,
                               out_channels=out_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        self.activation = activation_dict[activation]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch_size, input_dim, H, W, L)
        """
        x = self.deconv0(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.conv0(x)
        # x = self.deconv1(x)
        # x = self.activation(x)
        return x


class Interp3dUpsample(nn.Module):
    '''
    interpolate then Conv3dResBlock
    old code uses lambda and cannot be pickled
    temp hard-coded dimensions
    '''

    def __init__(self, in_dim: int,
                 out_dim: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 residual=False,
                 conv_block=True,
                 interp_mode='trilinear',
                 interp_size=None,
                 activation='silu',
                 dropout=0.1):
        super(Interp3dUpsample, self).__init__()
        self.activation = activation_dict[activation]
        self.dropout = nn.Dropout(dropout)
        if conv_block:
            self.conv = nn.Sequential(Conv3dResBlock(
                in_dim, out_dim,
                kernel_size=kernel_size,
                padding=padding,
                residual=residual,
                dropout=dropout,
                activation=activation),
                self.dropout,
                self.activation)
        self.conv_block = conv_block
        self.interp_size = interp_size
        self.interp_mode = interp_mode

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch_size, input_dim, H, W, L)
        """
        if self.conv_block:
            x = self.conv(x)

        x = F.interpolate(x, size=self.interp_size,
                          mode=self.interp_mode,
                          align_corners=True)
        return x


if __name__ == "__main__":
    x = torch.ones([10, 3, 64])
    layer = Conv1dResBlock(in_dim=3, out_dim=10, residual=True)
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 3, 64])
    layer = Interp1dUpsample(in_dim=3, out_dim=10, interp_size=[128])
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 3, 5])
    layer = DeConv1dBlock(in_dim=3, hidden_dim=32, out_dim=10)
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 3, 64, 66])
    layer = Conv2dResBlock(in_dim=3, out_dim=10, residual=True)
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 3, 64, 64])
    layer = Interp2dUpsample(in_dim=3, out_dim=10, interp_size=[128, 128])
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 3, 5, 5])
    layer = DeConv2dBlock(in_dim=3, hidden_dim=32, out_dim=10)
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 3, 64, 66, 66])
    layer = Conv3dResBlock(in_dim=3, out_dim=10, residual=True)
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 3, 64, 64, 66])
    layer = Interp3dUpsample(in_dim=3, out_dim=10, interp_size=[258, 258, 258])
    y = layer(x)
    print(y.shape)

    x = torch.ones([10, 3, 5, 5, 66])
    layer = DeConv3dBlock(in_dim=3, hidden_dim=32, out_dim=10)
    y = layer(x)
    print(y.shape)
