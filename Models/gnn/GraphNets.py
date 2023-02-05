#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/6 17:37
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : GraphNets.py
"""
import torch
from torch_geometric.nn.conv import GMMConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform
from torch_sparse import coalesce
from torch.nn import Parameter
import torch.nn.functional as F

from Models.configs import *
from basic_layers import Identity


class GMMResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, attr_dim,
                 kernel_size=3,
                 dropout=0.1,
                 residual=False,
                 activation='silu',
                 basic_block=False,
                 ):
        super(GMMResBlock, self).__init__()

        self.activation = activation_dict[activation]
        self.add_res = residual
        self.dropout = nn.Dropout(dropout)
        self.conv = GMMConv(in_dim, out_dim, dim=attr_dim, kernel_size=kernel_size)
        self.basic_block = basic_block
        if self.basic_block:
            self.conv1 = GMMConv(out_dim, out_dim, dim=attr_dim, kernel_size=kernel_size)
        self.apply_shortcut = (in_dim != out_dim)

        if self.add_res:
            if self.apply_shortcut:
                self.res = Identity(in_dim, out_dim)
            else:
                self.res = Identity()

    def forward(self, x, edge_index, edge_attr):
        """
        forward compute
        :param in_var: (batch_size, input_dim, H, W)
        """
        if self.add_res:
            h = self.res(x)

        x = self.conv(x, edge_index, edge_attr)
        x = self.dropout(x)

        if self.basic_block:
            x = self.activation(x)
            x = self.conv1(x, edge_index, edge_attr)
            x = self.dropout(x)

        if self.add_res:
            return self.activation(x + h)
        else:
            return self.activation(x)


class KernelNN3(torch.nn.Module):
    def __init__(self, width_node, width_kernel, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN3, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width_node)

        kernel1 = DenseNet([ker_in, width_kernel // 2, width_kernel, width_node ** 2], torch.nn.ReLU)
        self.conv1 = NNConvOld(width_node, width_node, kernel1, aggr='mean')
        kernel2 = DenseNet([ker_in, width_kernel // 2, width_kernel, width_node ** 2], torch.nn.ReLU)
        self.conv2 = NNConvOld(width_node, width_node, kernel2, aggr='mean')
        kernel3 = DenseNet([ker_in, width_kernel // 2, width_kernel, width_node ** 2], torch.nn.ReLU)
        self.conv3 = NNConvOld(width_node, width_node, kernel3, aggr='mean')
        kernel4 = DenseNet([ker_in, width_kernel // 2, width_kernel, width_node ** 2], torch.nn.ReLU)
        self.conv4 = NNConvOld(width_node, width_node, kernel4, aggr='mean')

        self.fc2 = torch.nn.Linear(width_node, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.conv4(x, edge_index, edge_attr)

        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        return x


class NNConvOld(MessagePassing):
    r"""The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    This convolution is also known as the edge-conditioned convolution from the
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
    :class:`torch_geometric.nn.conv.ECConv` for an alias):
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),
    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
    a MLP.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`edge_attr` of shape :obj:`[-1,
            num_edge_features]` to shape
            :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 aggr='add',
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(NNConvOld, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        size = self.in_channels
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_j, pseudo):
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


class GMMNet(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, width=32, depth=4, basic_block=None, activation='gelu', dropout=0.0):
        super(GMMNet, self).__init__()

        self.fc1 = torch.nn.Linear(in_dim, width)
        self.layers = nn.ModuleList()
        for i in range(depth):
            if basic_block is None:
                self.layers.append(GMMResBlock(width, width, attr_dim=edge_dim, residual=True,
                                               kernel_size=3, basic_block=True, activation=activation, dropout=dropout))
            else:
                self.layers.append(NNConvOld(width, width, attr_dim=edge_dim, residual=True,
                                             kernel_size=3, basic_block=True, activation=activation, dropout=dropout))

        self.fc2 = torch.nn.Linear(width, 128)
        self.fc3 = torch.nn.Linear(128, out_dim)
        self.activation = activation_dict[activation]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x, edge_index, edge_attr)
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class GMMNet_U(nn.Module):
    def __init__(self, in_sizes: tuple[int, ...], out_sizes: tuple[int, ...], width=32, depth=4, activation='gelu',
                 dropout=0.0):
        super(GMMNet_U, self).__init__()
        self.width = width
        self.depth = depth
        self.activation = activation
        self.dropout = dropout

        self.encoders = nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
                self.encoders.append(
                    GMMResBlock(self.in_dim, width, basic_block=True, activation=activation, dropout=dropout))
            else:
                self.encoders.append(
                    GMMResBlock(2 ** (i - 1) * width, 2 ** i * width, basic_block=True, activation=activation,
                                dropout=dropout))

        self.bottleneck = GMMResBlock(2 ** i * width, 2 ** i * width * 2, basic_block=True, activation=activation,
                                      dropout=dropout)

        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        for i in range(self.depth, 0, -1):
            self.decoders.append(
                GMMResBlock(2 ** i * width, 2 ** (i - 1) * width, activation=activation, dropout=dropout))
            self.upconvs.append(
                GMMResBlock(2 ** i * width, 2 ** (i - 1) * width, 2 ** (i - 1) * width, activation=activation,
                            dropout=dropout))

        self.conv1 = GMMResBlock(in_dim=width, out_dim=self.out_dim, basic_block=False, activation=activation,
                                 dropout=dropout)

        self.interp_out = Interp2dUpsample(in_dim=self.out_dim, out_dim=self.out_dim, interp_size=self.out_sizes,
                                           conv_block=False, activation=activation, dropout=dropout)

        self.conv2 = nn.GMMConv(self.out_dim, self.out_dim, dim=1, kernel_size=3)

    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = []
        for i in range(self.half_layers - 1):
            x = self.layers[i](x, edge_index, edge_attr)
            x = self.active(x)
            h.append(x)

        x = self.layers[self.half_layers - 1](x, edge_index, edge_attr)
        x = self.active(x)

        for i in range(self.half_layers, self.all_layers - 1):
            x = self.layers[i](x, edge_index, edge_attr)
            x += h[self.all_layers - 2 - i]
            x = self.active(x)

        x = self.layers[-1](x, edge_index, edge_attr)
        return x


if __name__ == '__main__':
    x = torch.ones([1000, 3])
    edge = torch.randint(0, x.shape[0], (2, x.shape[0] * 3))
    edge_index, edge_attr = coalesce(edge, None, x.shape[0], x.shape[0])
    edge_attr = x[edge_index.T].reshape((edge_index.shape[-1], -1))
    layer = GMMResBlock(x.shape[-1], 32, attr_dim=edge_attr.shape[-1], basic_block=True, activation='gelu', dropout=0.0)
    y = layer(x, edge_index, edge_attr)
    print(y.shape)

    Net_model = GMMNet(in_dim=2, out_dim=1, edge_dim=6, width=32, depth=4, activation='gelu')
    y = Net_model(x, edge_index, edge_attr)
    print(y.shape)
