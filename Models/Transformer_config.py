#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/9 0:13
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : Model_all.py
"""


from Models.Transformers import *

########################################## make full model　###########################################
config = defaultdict(lambda: None,
                     node_feats=259,  # node [batch_size, seq_len, feats]
                     edge_feats=0,
                     pos_dim=2,  #
                     n_targets=3,
                     n_hidden=128,  # node after feature extract [batch_size, seq_len, n_hidden/d_model]
                     num_feat_layers=2,  # feature extract layer number
                     num_encoder_layers=5,  # encoder layer number
                     encoder_dropout=0.0,
                     ffn_dropout=0.0,
                     attn_activation='gelu',
                     n_head=4,
                     pred_len=0,
                     n_freq_targets=0,
                     dim_feedforward=128 * 2,
                     feat_extract_type='Identity',  # feature extract
                     graph_activation=True,
                     raw_laplacian=True,
                     attention_type='galerkin',  # no softmax
                     xavier_init=1.0e-2,
                     diagonal_weight=1.0e-2,
                     symmetric_init=False,
                     layer_norm=True,
                     attn_norm=False,
                     batch_norm=False,
                     spacial_residual=False,
                     return_attn_weight=True,
                     seq_len=None,
                     bulk_regression=False,
                     decoder_type='pointwise',  # spectral ifft require n_grid * n_grid
                     decoder_dropout=0.0,
                     regressor_activation='gelu',
                     freq_dim=64,
                     num_regressor_layers=4,
                     fourier_modes=16,
                     spacial_dim=2,  # 1d / 2d Conv
                     spacial_fc=False,  # add the spacial_dim to in_dim( put the grid in)
                     dropout=0.0,
                     debug=False,
                     )


def get_transformer(name):

    if name=="lite":
        config['n_hidden'] = 128
        config['num_encoder_layers'] = 4
        config['n_head'] = 4

    elif name=="base":

        config['n_hidden'] = 256
        config['num_encoder_layers'] = 8
        config['n_head'] = 8
    elif name=="big":

        config['n_hidden'] = 512
        config['num_encoder_layers'] = 8
        config['n_head'] = 8

    elif name=="huge":
        config['n_hidden'] = 1024
        config['num_encoder_layers'] = 16
        config['n_head'] = 16

    return SimpleTransformer(**config)