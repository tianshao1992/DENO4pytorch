#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/7/17 0:40
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : data_normer.py
# @Description    : ******
"""

import os
import numpy as np
import pandas as pd
import torch
import pickle


class DataNormer(object):
    """
        data normalization at last dimension
    """

    def __init__(self, data, method="min-max", axis=None, epsilon=1e-10):
        """
            data normalization at last dimension
            :param data: data to be normalized
            :param method: normalization method
            :param axis: axis to be normalized
        """
        if isinstance(data, str):
            if os.path.isfile(data):
                try:
                    self.load(data)
                except:
                    raise ValueError("the savefile format is not supported!")
            else:
                raise ValueError("the file does not exist!")
        elif type(data) is np.ndarray:
            data = data
        elif type(data) is torch.Tensor:
            data = np.array(data.cpu(), dtype=np.float32)
        elif type(data) is pd.DataFrame:
            data = np.array(data.values, dtype=np.float32)
        else:
            raise NotImplementedError("the data type is not supported!")

        if axis is None:
            axis = tuple(range(len(data.shape) - 1))
        self.method = method
        if method == "min-max":
            self.max = np.max(data, axis=axis)
            self.min = np.min(data, axis=axis)
        elif method == "mean-std":
            self.mean = np.mean(data, axis=axis)
            self.std = np.std(data, axis=axis)

        self.epsilon = epsilon

    def norm(self, x):
        """
            input tensors
            param x: input tensors
            return x: output tensors
        """
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = 2 * (x - torch.tensor(self.min, device=x.device)) \
                    / (torch.tensor(self.max, device=x.device) - torch.tensor(self.min, device=x.device)
                       + self.epsilon) - 1
            elif self.method == "mean-std":
                x = (x - torch.tensor(self.mean, device=x.device)) \
                    / (torch.tensor(self.std + self.epsilon, device=x.device))
        else:
            if self.method == "min-max":
                x = 2 * (x - self.min) / (self.max - self.min + self.epsilon) - 1
            elif self.method == "mean-std":
                x = (x - self.mean) / (self.std + self.epsilon)

        return x

    def back(self, x):
        """
            input tensors
            param x: input tensors
            return x: output tensors
        """
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = (x + 1) / 2 * (torch.tensor(self.max, device=x.device)
                                   - torch.tensor(self.min, device=x.device) + self.epsilon) \
                    + torch.tensor(self.min, device=x.device)
            elif self.method == "mean-std":
                x = x * (torch.tensor(self.std + self.epsilon, device=x.device)) + torch.tensor(self.mean,
                                                                                                device=x.device)
        else:
            if self.method == "min-max":
                x = (x + 1) / 2 * (self.max - self.min + self.epsilon) + self.min
            elif self.method == "mean-std":
                x = x * (self.std + self.epsilon) + self.mean
        return x

    def save(self, save_path):
        """
            save the parameters to the file
            :param save_path: file path to save
        """

        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, save_path):
        """
            load the parameters from the file
            :param save_path: file path to load
        """

        isExist = os.path.exists(save_path)
        if isExist:
            try:
                with open(save_path, 'rb') as f:
                    load = pickle.load(f)
                self.method = load.method
                if load.method == "mean-std":
                    self.std = load.std
                    self.mean = load.mean
                elif load.method == "min-max":
                    self.min = load.min
                    self.max = load.max
            except:
                raise ValueError("the savefile format is not supported!")
        else:
            raise ValueError("The pkl file is not exist, CHECK PLEASE!")
