#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/6 17:37
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : graph_layers.py
"""

import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_

# from torch_geometric.nn import GCNConv
# from utilize import activation_dict

from cnn.conv_layers import Conv2dResBlock


class GraphConvolution(nn.Module):
    """
    A modified implementation from
    https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    to incorporate batch size, and multiple edge

    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, debug=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.debug = debug
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        layer weight initialized
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, edge):
        '''
        input: node feats (-1, seq_len, n_feats)
               edge only takes adj (-1, seq_len, seq_len)
               edge matrix first one in the last dim is graph Lap.
        '''
        if x.size(-1) != self.in_features:
            x = x.transpose(-2, -1).contiguous()
        assert x.size(1) == edge.size(-1)
        support = torch.matmul(x, self.weight)

        support = support.transpose(-2, -1).contiguous()
        output = torch.matmul(edge, support.unsqueeze(-1))

        output = output.squeeze()
        if self.bias is not None:
            return output + self.bias.unsqueeze(-1)
        else:
            return output

    def __repr__(self):
        """
        __repr__
        """
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttention(nn.Module):
    """
    Simple GAT layer, modified from https://github.com/Diego999/pyGAT/blob/master/layers.py
    to incorporate batch size similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features,
                 out_features,
                 alpha=1e-2,
                 concat=True,
                 graph_lap=True,  # graph laplacian may have negative entries
                 interaction_thresh=1e-6,
                 dropout=0.1):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.graph_lap = graph_lap
        self.thresh = interaction_thresh

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        xavier_normal_(self.W, gain=np.sqrt(2.0))

        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        xavier_normal_(self.a, gain=np.sqrt(2.0))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, node, adj):
        '''
        input: node feats (-1, seq_len, n_feats)
               edge only takes adj (-1, seq_len, seq_len)
               edge matrix first one in the last dim is graph Lap.
        '''
        h = torch.matmul(node, self.W)
        bsz, seq_len = h.size(0), h.size(1)

        a_input = torch.cat([h.repeat(1, 1, seq_len).view(bsz, seq_len * seq_len, -1),
                             h.repeat(1, seq_len, 1)], dim=2)
        a_input = a_input.view(bsz, seq_len, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        zero_vec = -9e15 * torch.ones_like(e)
        if self.graph_lap:
            attention = torch.where(adj.abs() > self.thresh, e, zero_vec)
        else:
            attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class EdgeEncoder(nn.Module):
    def __init__(self, out_dim: int,
                 edge_feats: int,
                 raw_laplacian=None):
        super(EdgeEncoder, self).__init__()
        assert out_dim > edge_feats
        self.return_lap = raw_laplacian
        if self.return_lap:
            out_dim = out_dim - edge_feats

        conv_dim0 = int(out_dim / 3 * 2)
        conv_dim1 = int(out_dim - conv_dim0)
        self.lap_conv1 = Conv2dResBlock(edge_feats, conv_dim0)
        self.lap_conv2 = Conv2dResBlock(conv_dim0, conv_dim1)

    def forward(self, lap):
        """
        forward compute
        :param lap: (batch_size, input_dim, H, W)
        """
        edge1 = self.lap_conv1(lap)
        edge2 = self.lap_conv2(edge1)
        if self.return_lap:
            return torch.cat([lap, edge1, edge2], dim=1)
        else:
            return torch.cat([edge1, edge2], dim=1)

    def __repr__(self):
        """
        __repr__
        """
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    '''
    A simple GCN, a wrapper for Kipf and Weiling's code
    learnable edge features similar to
    Graph Transformer https://arxiv.org/abs/1911.06455
    but using neighbor agg
    '''

    def __init__(self,
                 node_feats=4,
                 out_features=96,
                 num_gcn_layers=2,
                 edge_feats=6,
                 activation=True,
                 raw_laplacian=False,
                 dropout=0.1,
                 debug=False):
        super(GCN, self).__init__()

        self.edge_learner = EdgeEncoder(out_dim=out_features,
                                        edge_feats=edge_feats,
                                        raw_laplacian=raw_laplacian
                                        )
        self.gcn_layer0 = GraphConvolution(in_features=node_feats,  # hard coded
                                           out_features=out_features,
                                           debug=debug,
                                           )
        self.gcn_layers = nn.ModuleList([copy.deepcopy(GraphConvolution(
            in_features=out_features,  # hard coded
            out_features=out_features,
            debug=debug
        )) for _ in range(1, num_gcn_layers)])
        self.activation = activation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.edge_feats = edge_feats
        self.debug = debug

    def forward(self, x, edge):
        '''
        input: node feats (-1, seq_len, n_feats)
               edge only takes adj (-1, seq_len, seq_len)
               edge matrix first one in the last dim is graph Lap.
        '''
        x = x.permute(0, 2, 1).contiguous()
        edge = edge.permute([0, 3, 1, 2]).contiguous()
        assert edge.size(1) == self.edge_feats

        edge = self.edge_learner(edge)

        out = self.gcn_layer0(x, edge)
        for gc in self.gcn_layers[:-1]:
            out = gc(out, edge)
            if self.activation:
                out = self.relu(out)

        # last layer no activation
        out = self.gcn_layers[-1](out, edge)
        return out.permute(0, 2, 1)


class GAT(nn.Module):
    '''
    A simple GAT: modified from the official implementation
    '''

    def __init__(self,
                 node_feats=4,
                 out_features=96,
                 num_gcn_layers=2,
                 edge_feats=None,
                 activation=False,
                 debug=False):
        super(GAT, self).__init__()

        self.gat_layer0 = GraphAttention(in_features=node_feats,
                                         out_features=out_features,
                                         )
        self.gat_layers = nn.ModuleList([copy.deepcopy(GraphAttention(
            in_features=out_features,
            out_features=out_features,
        )) for _ in range(1, num_gcn_layers)])
        self.activation = activation
        self.relu = nn.ReLU()
        self.debug = debug

    def forward(self, x, edge):
        '''
        input: node feats (-1, seq_len, n_feats)
               edge only takes adj (-1, seq_len, seq_len)
               edge matrix first one in the last dim is graph Lap.
        '''
        edge = edge[..., 0].contiguous()

        out = self.gat_layer0(x, edge)

        for layer in self.gat_layers[:-1]:
            out = layer(out, edge)
            if self.activation:
                out = self.relu(out)

        # last layer no activation
        return self.gat_layers[-1](out, edge)
