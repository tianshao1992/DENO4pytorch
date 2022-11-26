import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn.conv import SplineConv, GMMConv, GATConv, SAGEConv, GCNConv, PANConv
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_sparse import coalesce

from Models.configs import *
from basic_layers import Identity


class GMMResBlock(nn.Module):
    def __init__(self, in_dim, out_dim,
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
        self.conv = GMMConv(out_dim, out_dim, dim=1, kernel_size=kernel_size)
        self.basic_block = basic_block
        if self.basic_block:
            self.conv1 = GMMConv(out_dim, out_dim, dim=1, kernel_size=kernel_size)
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


class GMMNet(nn.Module):
    def __init__(self, planes):
        super(GMMNet, self).__init__()
        self.planes = planes
        self.layers = nn.ModuleList()
        for i in range(len(self.planes) - 2):
            self.layers.append(GMMResBlock(self.planes[i], self.planes[i + 1], kernel_size=3))

        self.layers.append(GMMResBlock(self.planes[-2], self.planes[-1], kernel_size=3))
        self.active = nn.GELU()

    def forward(self, x, edge_index, edge_attr):

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, edge_index, edge_attr)
            x = self.active(x)
        x = self.layers[-1](x, edge_index, edge_attr)
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
    layer = GMMResBlock(x.shape[-1], 32, basic_block=True, activation='gelu', dropout=0.0)
    y = layer(x, edge_index, edge_attr)
    print(y.shape)
