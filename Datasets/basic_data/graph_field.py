#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/6/8 11:04
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : graph_field.py
"""
import torch
import numpy as np
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict
from Datasets.basic_data._base_data import BasicData
from collections import namedtuple

IndexType = Union[slice, torch.Tensor, np.ndarray, Sequence]
DataType = namedtuple('GraphType', ['data', 'mesh', 'edge'])


class GraphField(BasicData):
    """Graph Field base class."""

    def __init__(self,
                 data: Union[np.ndarray, torch.Tensor],
                 mesh,
                 edge=None,
                 name=None):
        """
         Args:
            data: numpy.ndarray, nd = 3
                  1st dimension for sample
                  2nd dimension for point
                  last dimension for property
            mesh:
                  numpy.ndarray, nd = 3
                  1st dimension for sample
                  2nd dimension for point
                  last dimension for space
            edge:
                  numpy.ndarray, nd = 2
            name: list of str to describe the property
        """
        self.data = data
        self.name = name
        self.mesh = mesh
        self.edge = edge

        self.sample_size = data.shape[0]
        self.points_size = data.shape[1]
        self.proper_size = data.shape[2]

        self.name = self._get_name()

    def __getitem__(self, idx):
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
            data = np.array(self.data[idx], dtype=np.float32)
            mesh = np.array(self.mesh[idx], dtype=np.float32)
            edge = np.array(self.edge[idx], dtype=np.long)
            return DataType(data=data, mesh=mesh, edge=edge)
        else:
            idx = self._select_index(idx)
            data = self.data[idx]
            mesh = self.mesh[idx]
            edge = self.edge[idx]
        return GraphField(data=data, mesh=mesh, edge=edge, name=self.name)

    def get_edge(self):
        """
        get the grid array of the GraphField

        Args:

        Returns:
            numpy ndarray
        """
        pass

    def load_fromfile(self):
        """
        load data and grid from file
        Args:
        Returns:
            None
        """
        raise NotImplementedError("")
