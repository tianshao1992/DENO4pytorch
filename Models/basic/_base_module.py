#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/7/16 22:47
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : basic_models.py
# @Description    : ******
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/6/8 13:05
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : _base_module.py
"""


import abc
import numpy as np
import pandas as pd
# import taichi as ti
import torch
# import paddle

class BasicModule(abc.ABC):

    """Module base class."""


    def save_model(self, file_path: str):
        """
        save a BaseRegression to a file.
        Any non-abstract model inherited from this class should implement this method.

        Args:
            file_path(str): The path of the file to save to
            file_type(str): The type of the file to save to

        Returns:
            None
        """
        pass

    def train(self, train_dataset, valid_dataset, lossfunc, optimizer, scheduler, device):
        """
        train a BaseRegression instance.
        Any non-abstract model inherited from this class should implement this method.

        Args:
            train_dataset(BaseDataset): Train set, including the input and label.
            valid_dataset(Dataset|None): Eval set, used for early stopping.
            lossfunc: Loss function in the training process
            optimizer: Optimizer in the training process
            scheduler: Learning rate scheduler of the optimizer in the training process
            device(Device|None): default as None in cpu device.
        """
        pass

    def infer(self, data, device):
        """
        inference a BaseRegression instance.

        Any non-abstract model inherited from this class should implement this method.

        Args:
            data(BaseDataset): Train set, including the input and label.
            device(Device|None): default as None in cpu device.
        """
        pass

    def _train_step(self, train_dataset, valid_dataset, lossfunc, optimizer, scheduler, device):

        """
        train a BaseRegression instance.
        Any non-abstract model inherited from this class should implement this method.

        Args:
            train_dataset(BaseDataset): Train set, including the input and label.
            valid_dataset(Dataset|None): Eval set, used for early stopping.
            lossfunc: Loss function in the training process
            optimizer: Optimizer in the training process
            scheduler: Learning rate scheduler of the optimizer in the training process
            device(Device|None): default as None in cpu device.
        """
        pass

    def _valid_step(self, valid_dataset, lossfunc, device):
        """
        valid a BaseRegression instance.
        Any non-abstract model inherited from this class should implement this method.

        Args:
            valid_dataset(Dataset|None): Eval set, used for early stopping.
            lossfunc: Loss function in the training process
            device(Device|None): default as None in cpu device.
        """
        pass


    def _set_lossfunc(self, lossfunc):
        """
        set loss function for the model.
        Args:
            lossfunc: Loss function in the training process
        """
        self.lossfunc = lossfunc


    def _set_optimizer(self, optimizer):
        """
        set optimizer for the model.

        Args:
            optimizer: Optimizer in the training process
        """
        self.optimizer = optimizer

    def _set_scheduler(self, scheduler):
        """
        set scheduler for the model.

        Args:
            scheduler: scheduler in the training process
        """
        self.scheduler = scheduler