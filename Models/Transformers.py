#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/6 17:37
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : Transformers.py
"""

import os
import copy
import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torchinfo import summary

from collections import defaultdict
from functools import partial

from GraphTransformer.utilize import *
from GraphTransformer.basic_layers import *
from GraphTransformer.graph_layers import *
from GraphTransformer.attention_layers import *
from GraphTransformer.spectral_layers import *


class SimpleTransformerEncoderLayer(nn.Module):
    """
    EncoderLayer for transformer
    """

    def __init__(self,
                 d_model=96,
                 pos_dim=1,
                 n_head=2,
                 dim_feedforward=512,
                 attention_type='fourier',
                 pos_emb=False,
                 layer_norm=True,
                 attn_norm=None,
                 norm_type='layer',
                 norm_eps=None,
                 batch_norm=False,
                 attn_weight=False,
                 xavier_init: float = 1e-2,
                 diagonal_weight: float = 1e-2,
                 symmetric_init=False,
                 residual_type='add',
                 activation_type='relu',
                 dropout=0.1,
                 ffn_dropout=None,
                 debug=False,
                 ):
        super(SimpleTransformerEncoderLayer, self).__init__()

        dropout = default(dropout, 0.05)
        if attention_type in ['linear', 'softmax']:
            dropout = 0.1
        ffn_dropout = default(ffn_dropout, dropout)
        norm_eps = default(norm_eps, 1e-5)
        attn_norm = default(attn_norm, not layer_norm)
        if (not layer_norm) and (not attn_norm):
            attn_norm = True
        norm_type = default(norm_type, 'layer')

        self.attn = SimpleAttention(n_head=n_head,
                                    d_model=d_model,
                                    attention_type=attention_type,
                                    diagonal_weight=diagonal_weight,
                                    xavier_init=xavier_init,
                                    symmetric_init=symmetric_init,
                                    pos_dim=pos_dim,
                                    norm_add=attn_norm,
                                    norm_type=norm_type,
                                    eps=norm_eps,
                                    dropout=dropout)
        self.d_model = d_model
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.add_layer_norm = layer_norm
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(d_model, eps=norm_eps)
            self.layer_norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        dim_feedforward = default(dim_feedforward, 2 * d_model)
        self.ff = FeedForward(in_dim=d_model,
                              dim_feedforward=dim_feedforward,
                              batch_norm=batch_norm,
                              activation=activation_type,
                              dropout=ffn_dropout,
                              )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_type = residual_type  # plus or minus
        self.add_pos_emb = pos_emb
        if self.add_pos_emb:
            self.pos_emb = PositionalEncoding(d_model)

        self.debug = debug
        self.attn_weight = attn_weight
        self.__name__ = attention_type.capitalize() + 'TransformerEncoderLayer'

    def forward(self, x, pos=None, weight=None):
        '''
        - x: node feature, (batch_size, seq_len, n_feats)
        - pos: position coords, needed in every head

        Remark:
            - for n_head=1, no need to encode positional
            information if coords are in features
        '''
        if self.add_pos_emb:
            x = x.permute((1, 0, 2))
            x = self.pos_emb(x)
            x = x.permute((1, 0, 2))

        if pos is not None and self.pos_dim > 0:
            att_output, attn_weight = self.attn(
                x, x, x, pos=pos, weight=weight)  # encoder no mask
        else:
            att_output, attn_weight = self.attn(x, x, x, weight=weight)

        if self.residual_type in ['add', 'plus'] or self.residual_type is None:
            x = x + self.dropout1(att_output)
        else:
            x = x - self.dropout1(att_output)
        if self.add_layer_norm:
            x = self.layer_norm1(x)

        x1 = self.ff(x)
        x = x + self.dropout2(x1)

        if self.add_layer_norm:
            x = self.layer_norm2(x)

        if self.attn_weight:
            return x, attn_weight
        else:
            return x


class PointwiseRegressor(nn.Module):
    '''
    A wrapper for a simple pointwise linear layers
    '''

    def __init__(self, in_dim,  # input dimension
                 n_hidden,
                 out_dim,  # number of target dim
                 num_layers: int = 2,
                 spacial_fc: bool = False,
                 spacial_dim=1,
                 dropout=0.1,
                 activation='silu',
                 return_latent=False,
                 debug=False):
        super(PointwiseRegressor, self).__init__()

        dropout = default(dropout, 0.1)
        self.spacial_fc = spacial_fc
        activ = activation_dict[activation]
        if self.spacial_fc:
            in_dim = in_dim + spacial_dim
            self.fc = nn.Linear(in_dim, n_hidden)
        self.ff = nn.ModuleList([nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            activ,
        )])
        for _ in range(num_layers - 1):
            self.ff.append(nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                activ,
            ))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_hidden, out_dim)
        self.return_latent = return_latent
        self.debug = debug

    def forward(self, x, grid=None):
        '''
        2D:
            Input: (-1, n, n, in_features)
            Output: (-1, n, n, n_targets)
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        '''
        if self.spacial_fc:
            x = torch.cat([x, grid], dim=-1)
            x = self.fc(x)

        for layer in self.ff:
            x = layer(x)
            x = self.dropout(x)

        x = self.out(x)

        if self.return_latent:
            return x, None
        else:
            return x


class SpectralRegressor(nn.Module):
    '''
    A wrapper for both SpectralConv1d and SpectralConv2d
    Ref: Li et 2020 FNO paper
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
    A new implementation incoporating all spacial-based FNO
    in_dim: input dimension, (either n_hidden or spacial dim)
    n_hidden: number of hidden features out from attention to the fourier conv
    '''

    def __init__(self, in_dim,
                 n_hidden,
                 freq_dim,
                 out_dim,
                 modes: int,
                 num_spectral_layers: int = 2,
                 n_grid=None,
                 dim_feedforward=None,
                 spacial_fc=False,
                 spacial_dim=2,
                 return_freq=False,
                 return_latent=False,
                 normalizer=None,
                 activation='silu',
                 last_activation=True,
                 dropout=0.1,
                 debug=False):
        super(SpectralRegressor, self).__init__()
        if spacial_dim == 2:  # 2d, function + (x,y)
            spectral_conv = SpectralConv2d
        elif spacial_dim == 1:  # 1d, function + x
            spectral_conv = SpectralConv1d
        else:
            raise NotImplementedError("3D not implemented.")
        # activation = default(activation, 'silu')
        self.activation = activation_dict[activation]
        dropout = default(dropout, 0.1)
        self.spacial_fc = spacial_fc  # False in Transformer
        if self.spacial_fc:
            self.fc = nn.Linear(in_dim + spacial_dim, n_hidden)
        self.spectral_conv = nn.ModuleList([spectral_conv(in_dim=n_hidden,
                                                          out_dim=freq_dim,
                                                          n_grid=n_grid,
                                                          modes=modes,
                                                          dropout=dropout,
                                                          activation=activation,
                                                          return_freq=return_freq,
                                                          debug=debug)])
        for _ in range(num_spectral_layers - 1):
            self.spectral_conv.append(spectral_conv(in_dim=freq_dim,
                                                    out_dim=freq_dim,
                                                    n_grid=n_grid,
                                                    modes=modes,
                                                    dropout=dropout,
                                                    activation=activation,
                                                    return_freq=return_freq,
                                                    debug=debug))
        if not last_activation:
            self.spectral_conv[-1].activation = Identity()

        self.n_grid = n_grid  # dummy for debug
        self.dim_feedforward = default(dim_feedforward, 2 * spacial_dim * freq_dim)
        self.regressor = nn.Sequential(
            nn.Linear(freq_dim, self.dim_feedforward),
            self.activation,
            nn.Linear(self.dim_feedforward, out_dim),
        )
        self.normalizer = normalizer
        self.return_freq = return_freq
        self.return_latent = return_latent
        self.debug = debug

    def forward(self, x, edge=None, pos=None, grid=None):
        '''
        2D:
            Input: (-1, n, n, in_features)
            Output: (-1, n, n, n_targets)
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        '''
        x_latent = []
        x_fts = []

        if self.spacial_fc:
            x = torch.cat([x, grid], dim=-1)
            x = self.fc(x)

        for layer in self.spectral_conv:
            if self.return_freq:
                x, x_ft = layer(x)
                x_fts.append(x_ft.contiguous())
            else:
                x = layer(x)

            if self.return_latent:
                x_latent.append(x.contiguous())

        x = self.regressor(x)

        if self.normalizer:
            x = self.normalizer.inverse_transform(x)

        if self.return_freq or self.return_latent:
            return x, dict(preds_freq=x_fts, preds_latent=x_latent)
        else:
            return x


class BulkRegressor(nn.Module):
    '''
    Bulk regressor:

    Args:
        - in_dim: seq_len
        - n_feats: pointwise hidden features
        - n_targets: number of overall bulk targets
        - pred_len: number of output sequence length
            in each sequence in each feature dimension (for eig prob this=1)

    Input:
        (-1, seq_len, n_features)
    Output:
        (-1, pred_len, n_target)
    '''

    def __init__(self, in_dim,  # seq_len
                 n_feats,  # number of hidden features
                 n_targets,  # number of frequency target
                 pred_len,
                 n_hidden=None,
                 sort_output=False,
                 dropout=0.1):
        super(BulkRegressor, self).__init__()
        n_hidden = default(n_hidden, pred_len * 4)
        self.linear = nn.Linear(n_feats, n_targets)
        freq_out = nn.Sequential(
            nn.Linear(in_dim, n_hidden),
            nn.LeakyReLU(),  # frequency can be localized
            nn.Linear(n_hidden, pred_len),
        )
        self.regressor = nn.ModuleList(
            [copy.deepcopy(freq_out) for _ in range(n_targets)])
        self.dropout = nn.Dropout(dropout)
        self.sort_output = sort_output

    def forward(self, x):
        """
        forward compute
        :param query: (batch, seq_len, d_model)
        """
        x = self.linear(x)
        x = x.transpose(-2, -1).contiguous()
        out = []
        for i, layer in enumerate(self.regressor):
            out.append(layer(x[:, i, :]))  # i-th target predict
        x = torch.stack(out, dim=-1)
        x = self.dropout(x)
        if self.sort_output:
            x, _ = torch.sort(x)
        return x


class SimpleTransformer(nn.Module):
    """
    simpleTransformers
    https://github.com/scaomath/galerkin-transformer
    """

    def __init__(self, **kwargs):
        super(SimpleTransformer, self).__init__()
        self.config = defaultdict(lambda: None, **kwargs)
        self._get_setting()
        self._initialize()
        self.__name__ = self.attention_type.capitalize() + 'Transformer'

    def forward(self, node, pos, edge=None, grid=None, weight=None):
        '''
        seq_len: n, number of grid points
        node_feats: number of features of the inputs
        edge_feats: number of Laplacian matrices (including learned)
        pos_dim: dimension of the Euclidean space
        - node: (batch_size, seq_len, node_feats)
        - pos: (batch_size, seq_len, pos_dim)
        - edge: (batch_size, seq_len, seq_len, edge_feats)
        - weight: (batch_size, seq_len, seq_len): mass matrix prefered
            or (batch_size, seq_len) when mass matrices are not provided

        Remark:
        for classic Transformer: pos_dim = n_hidden = 512
        pos encodings is added to the latent representation
        '''
        x_latent = []
        attn_weights = []

        x = self.feat_extract(node, edge)

        if self.spacial_residual or self.return_latent:
            res = x.contiguous()
            x_latent.append(res)

        for encoder in self.encoder_layers:
            if self.return_attn_weight:
                x, attn_weight = encoder(x, pos, weight)
                attn_weights.append(attn_weight)
            else:
                x = encoder(x, pos, weight)

            if self.return_latent:
                x_latent.append(x.contiguous())

        if self.spacial_residual:
            x = res + x

        x_freq = self.freq_regressor(
            x)[:, :self.pred_len, :] if self.n_freq_targets > 0 else None

        x = self.dpo(x)
        x = self.regressor(x, grid=grid)

        return dict(preds=x,
                    preds_freq=x_freq,
                    preds_latent=x_latent,
                    attn_weights=attn_weights)

    def _initialize(self):
        """
        parameter initialize
        """
        self._get_feature()

        self._get_encoder()

        if self.n_freq_targets > 0:
            self._get_freq_regressor()

        self._get_regressor()

        if self.decoder_type in ['pointwise', 'convolution']:
            self._initialize_layer(self.regressor)

        self.config = dict(self.config)

    @staticmethod
    def _initialize_layer(layer, gain=1e-2):
        """
        layer weight initialize
        """
        for param in layer.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=gain)
            else:
                constant_(param, 0)

    def _get_setting(self):
        """
        _get_setting
        """
        all_attr = list(self.config.keys()) + additional_attr
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.dim_feedforward = default(self.dim_feedforward, 2 * self.n_hidden)
        self.spacial_dim = default(self.spacial_dim, self.pos_dim)
        self.spacial_fc = default(self.spacial_fc, False)
        self.dropout = default(self.dropout, 0.05)
        self.dpo = nn.Dropout(self.dropout)
        if self.decoder_type == 'attention':
            self.num_encoder_layers += 1
        self.attention_types = ['fourier', 'integral',
                                'cosine', 'galerkin', 'linear', 'softmax']

    def _get_feature(self):
        """
        _get_feature
        """
        if self.num_feat_layers > 0 and self.feat_extract_type == 'gcn':
            self.feat_extract = GCN(node_feats=self.node_feats,
                                    edge_feats=self.edge_feats,
                                    num_gcn_layers=self.num_feat_layers,
                                    out_features=self.n_hidden,
                                    activation=self.graph_activation,
                                    raw_laplacian=self.raw_laplacian,
                                    debug=self.debug,
                                    )
        elif self.num_feat_layers > 0 and self.feat_extract_type == 'gat':
            self.feat_extract = GAT(node_feats=self.node_feats,
                                    out_features=self.n_hidden,
                                    num_gcn_layers=self.num_feat_layers,
                                    activation=self.graph_activation,
                                    debug=self.debug,
                                    )
        else:
            self.feat_extract = Identity(in_features=self.node_feats,
                                         out_features=self.n_hidden)

    def _get_encoder(self):
        """
        _get_encoder
        """
        if self.attention_type in self.attention_types:
            encoder_layer = SimpleTransformerEncoderLayer(d_model=self.n_hidden,
                                                          n_head=self.n_head,
                                                          attention_type=self.attention_type,
                                                          dim_feedforward=self.dim_feedforward,
                                                          layer_norm=self.layer_norm,
                                                          attn_norm=self.attn_norm,
                                                          norm_type=self.norm_type,
                                                          batch_norm=self.batch_norm,
                                                          pos_dim=self.pos_dim,
                                                          xavier_init=self.xavier_init,
                                                          diagonal_weight=self.diagonal_weight,
                                                          symmetric_init=self.symmetric_init,
                                                          attn_weight=self.return_attn_weight,
                                                          residual_type=self.residual_type,
                                                          activation_type=self.attn_activation,
                                                          dropout=self.encoder_dropout,
                                                          ffn_dropout=self.ffn_dropout,
                                                          debug=self.debug)
        # else:
        #     encoder_layer = _TransformerEncoderLayer(d_model=self.n_hidden,
        #                                              nhead=self.n_head,
        #                                              dim_feedforward=self.dim_feedforward,
        #                                              layer_norm=self.layer_norm,
        #                                              attn_weight=self.return_attn_weight,
        #                                              dropout=self.encoder_dropout
        #                                              )

        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])

    def _get_freq_regressor(self):
        """
        get_freq_regressor
        """
        if self.bulk_regression:
            self.freq_regressor = BulkRegressor(in_dim=self.seq_len,
                                                n_feats=self.n_hidden,
                                                n_targets=self.n_freq_targets,
                                                pred_len=self.pred_len)
        else:
            self.freq_regressor = nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_freq_targets),
            )

    def _get_regressor(self):
        """
        get_regressor
        """
        if self.decoder_type == 'pointwise':
            self.regressor = PointwiseRegressor(in_dim=self.n_hidden,
                                                n_hidden=self.n_hidden,
                                                out_dim=self.n_targets,
                                                spacial_fc=self.spacial_fc,
                                                spacial_dim=self.spacial_dim,
                                                activation=self.regressor_activation,
                                                dropout=self.decoder_dropout,
                                                debug=self.debug)
        elif self.decoder_type == 'ifft':
            self.regressor = SpectralRegressor(in_dim=self.n_hidden,
                                               n_hidden=self.n_hidden,
                                               freq_dim=self.freq_dim,
                                               out_dim=self.n_targets,
                                               num_spectral_layers=self.num_regressor_layers,
                                               modes=self.fourier_modes,
                                               spacial_dim=self.spacial_dim,
                                               spacial_fc=self.spacial_fc,
                                               dim_feedforward=self.freq_dim,
                                               activation=self.regressor_activation,
                                               dropout=self.decoder_dropout,
                                               )
        else:
            raise NotImplementedError("Decoder type not implemented")

    def get_graph(self):
        """
        get_graph
        """
        return self.gragh

    def get_encoder(self):
        """
        get_encoder
        """
        return self.encoder_layers


if __name__ == '__main__':
    for graph in ['gcn', 'gat']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = defaultdict(lambda: None,
                             node_feats=1,
                             edge_feats=5,
                             pos_dim=1,
                             n_targets=1,
                             n_hidden=96,
                             num_feat_layers=2,
                             num_encoder_layers=2,
                             n_head=2,
                             pred_len=0,
                             n_freq_targets=0,
                             dim_feedforward=96 * 2,
                             feat_extract_type=graph,
                             graph_activation=True,
                             raw_laplacian=True,
                             attention_type='fourier',  # no softmax
                             xavier_init=1e-4,
                             diagonal_weight=1e-2,
                             symmetric_init=False,
                             layer_norm=True,
                             attn_norm=False,
                             batch_norm=False,
                             spacial_residual=False,
                             return_attn_weight=True,
                             seq_len=None,
                             bulk_regression=False,
                             decoder_type='ifft',
                             freq_dim=64,
                             num_regressor_layers=2,
                             fourier_modes=16,
                             spacial_dim=1,
                             spacial_fc=True,
                             dropout=0.1,
                             debug=False,
                             )

        ft = SimpleTransformer(**config)
        ft.to(device)
        batch_size, seq_len = 8, 512
        summary(ft, input_size=[(batch_size, seq_len, 1),
                                (batch_size, seq_len, seq_len, 5),
                                (batch_size, seq_len, 1),
                                (batch_size, seq_len, 1)], device=device)

    # layer = TransformerEncoderLayer(d_model=128, nhead=4)
    # print(layer.__class__)
