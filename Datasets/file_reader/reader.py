#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/7/16 19:23
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : reader.py
# @Description    : ******
"""
import os.path
import numpy as np
import scipy
import scipy.io as sio
import torch
import h5py


# reading data
class DataReader(object):
    """
    读取mat文件
        params: file_path
        return: data
    """

    def __init__(self, file_path, to_torch=True, to_float=True, to_cuda=False):
        super(DataReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.load_file(file_path)

    def load_file(self, file_path):
        """
        加载文件，调用self._load_file
        """
        self.file_path = file_path
        self._load_file()

    def read_data(self, field):
        """
        读取field
        """
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        """
        设置是否使用cuda
        """
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        """
        设置是否使用torch
        """
        self.to_torch = to_torch

    def set_float(self, to_float):
        """
        设置是否使用float
        """
        self.to_float = to_float

    def _load_file(self):
        """
        加载文件: 目前支持格式: .mat, .npz, .npy, .pth
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError("File not found: {}".format(self.file_path))

        if self.file_path.endswith('.mat'):
            try:
                self.data = sio.loadmat(self.file_path)
                self.old_mat = True
            except:
                self.data = h5py.File(self.file_path)
                self.old_mat = False
        elif self.file_path.endswith('.npz'):
            self.data = np.load(self.file_path)
        elif self.file_path.endswith('.npy'):
            self.data = np.load(self.file_path)
        elif self.file_path.endswith('.pth'):
            self.data = torch.load(self.file_path)
        else:
            raise NotImplementedError("File type not supported: {}".format(self.file_path))
