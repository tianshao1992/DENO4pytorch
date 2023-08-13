#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/6/8 11:01
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : image_field.py
"""
import copy
import numpy as np
import torch
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict
from Datasets.basic_data._base_data import BasicData
from collections import namedtuple

IndexType = Union[slice, torch.Tensor, np.ndarray, Sequence]
DataType = namedtuple('ImageType', ['data', 'grid'])


class ImageField(BasicData):
    """ImageField base class."""

    def __init__(self,
                 data: Union[np.ndarray, torch.Tensor],
                 grid=None,
                 name=None):
        """
         Args:
            data: numpy.ndarray, nd >= 3
                  1st dimension sample
                  last dimension property
                  2nd - (n-1) grid shape
            grid:
                  numpy.ndarray, nd >= 3
                  1st dimension sample
                  last dimension space
                  2nd - (n-1) grid shape
            name: list of str to describe the property
        """
        self.data = data
        self.name = name
        self.grid = grid

        self.proper_size = data.shape[-1]
        self.sample_size = data.shape[0]
        self.coords_size = len(data.shape[1:-1])

        self.grid_shape = data.shape[1:-1]
        self.name = self._get_name()
        self.grid = self._get_grid()

    def _get_grid(self):
        """
        get the grid array of the ImageField
        Args:
        Returns:
            numpy ndarray
        """
        if self.grid is None:
            """generate a uniform mesh grid of the field
                refer：https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html """
            # todo: 为所有数组设置统一格式
            n_x = [np.linspace(0, 1, i).astype(np.float32) for i in self.grid_shape]
            self.grid = np.stack(np.meshgrid(*n_x), axis=-1)[None, ...]
            self._exist_grid = False
        else:
            self._check_grid()
            self._exist_grid = True
            return self.grid

    def visual_matplot(self, idx):
        """
        visual one sample by matplotlib
        Args:
        Returns:
            None
        """
        raise NotImplementedError("")

    def from_file(self, file_path, field_name):
        """
        load data and grid from file
        Args:
        Returns:
            None
        """

        # from Datasets.file_reader.reader import DataReader
        #
        # data_reader = DataReader(file_path, )
        # try:
        #     self.data = data_reader.read_data(field_name)
        # except:
        #     raise ValueError("field not found in file")
        # try:
        #     self.grid = data_reader.read_data('grid')
        # except:
        raise ImportWarning("grid not found in file")

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
            if self._exist_grid:
                grid = self.grid[idx]
            else:
                grid = self.grid
            return DataType(data=data, grid=grid)
        else:
            idx = self._select_index(idx)
            data = self.data[idx]
            if self._exist_grid:
                grid = self.grid[idx]
            else:
                grid = self.grid
            return ImageField(data, grid, name=self.name)

    def _check_grid(self):
        """
        check the grid and field dataformat
        """
        if self.grid.shape[0:-1] != self.data.shape[0:-1]:
            raise ValueError("grid shape must be equal to field shape!")


if __name__ == "__main__":
    fields = np.ones((100, 50, 70, 60, 6), dtype=np.float32)
    a = ImageField(fields)

    print(a.grid)
