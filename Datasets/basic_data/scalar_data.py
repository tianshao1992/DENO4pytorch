#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/6/8 10:33
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : basic_data.py
"""

import torch
import numpy as np
import pandas as pd
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict
from Datasets.basic_data._base_data import BasicData
from collections import namedtuple

IndexType = Union[slice, torch.Tensor, np.ndarray, Sequence]
DataType = namedtuple('ScalarType', ['data'])


class ScalarData(BasicData):
    r"""Data Scalar base class."""

    def __init__(self,
                 data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
                 name=None):
        """
         Args:
            data: numpy.ndarray, nd = 2
                  1st dimension sample
                  2nd dimension property
            name: list of str to describe the property
        """

        if isinstance(data, np.ndarray):
            if np.ndim(data) != 2:
                raise ValueError("The ndim of param 'data' must be 2")
            data = np.array(data, dtype=np.float32)
            self.name = name
        elif isinstance(data, pd.DataFrame):
            # when the type of data is dataframe, the name should be set as data.columns
            self.name = [str(i) for i in data.columns]
            data = np.array(data.values, dtype=np.float32)
        elif isinstance(data, torch.Tensor):
            data = np.array(data.cpu(), dtype=np.float32)
            # when the type of data is dataframe, the name should be set as data.columns
            self.name = name
        else:
            raise TypeError("The type of param `data` must be pd.DataFrame or np.ndarray or torch.Tensor,"
                            " but {type(data)} received")

        # data of ScalarData
        self.data = data

        # size of ScalarData
        self.proper_size = data.shape[1]
        self.sample_size = data.shape[0]

        # name of ScalarData
        self.name = self._get_name()

    def __getitem__(self,
                    idx: Union[int, np.integer, IndexType]):

        r"""
            In case :obj:`idx` is of type integer, will return the data object
            at index :obj:`idx` (and transforms it in case :obj:`transform` is present).
            In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
            tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type long or
            bool, will return a subset of the dataset at the specified indices.
        """
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, torch.Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            data = self.data[idx]
            return DataType(data=data)
        else:
            idx = self._select_index(idx)
            # dataset = copy.copy(self)
            # dataset.data = self.data[idx]
            return ScalarData(self.data[idx], self.name)

    def visual_matplot(self, idx):
        """
        visual one sample by matplotlib
        Args:
        Returns:
            None
        """
        raise NotImplementedError("")

    def from_file(self, file_path: str, file_type: str):
        """
        load data and grid from file
        Args:
        Returns:
            None
        """
        raise NotImplementedError("")


if __name__ == "__main__":
    a = ScalarData(data=np.ones((10, 5)))

    b = ScalarData(data=pd.DataFrame(np.ones((10, 5))))

    c = b[5:7]
    print(c)

    d = b[5]
    print(d)
