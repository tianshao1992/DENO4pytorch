#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/6/8 11:01
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : TimeSeries.py
"""

import torch
import numpy as np
import pandas as pd
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict

from Datasets.basic_data._base_data import BasicData

class TimeSeries(BasicData):
    """Data TimeSeries base class."""

    def __init__(self,
                 data: Union[pd.DataFrame, np.ndarray],
                 freq: Union[int, str],
                 name=None):
        """
         Args:
            data: numpy.ndarray, nd = 2
                  1st dimension time
                  2nd dimension property
            freq:
            name: list of str to describe the property
        """

        if isinstance(data, np.ndarray):
            if np.ndim(data) != 2:
                raise ValueError("The ndim of param 'data' must be 2")
            data = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            data = data
            # when the type of data is dataframe, the name should be set as data.columns
            name = [str(i) for i in data.columns]
        elif isinstance(data, torch.Tensor):
            data = data.numpy()
            data = pd.DataFrame(data)
            name = [str(i) for i in data.columns]
        else:
            raise TypeError("The type of param `data` must be pd.DataFrame or np.ndarray,"
                            " but {type(data)} received")

        # data of ScalarData
        self.data = data
        self.freq = freq

        # size of TimeSeries
        self.proper_size = data.shape[1]
        self.series_size = data.shape[0]

        # name of TimeSeries
        self.name = self._get_name(name)
        self.data.columns = self.name

    def __getitem__(self, index):

        return TimeSeries(self.data.iloc[index], self.freq, self.name)

    @property
    def time_index(self):
        """the time index"""
        return self.data.index





