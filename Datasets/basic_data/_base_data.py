#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/6/8 11:07
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : _base_data.py
"""

import abc
import copy
import numpy as np
import pandas as pd
# import taichi as ti
import torch
# import paddle
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict

IndexType = Union[slice, torch.Tensor, np.ndarray, Sequence]


class BasicData(abc.ABC):
    """Data Scalar base class."""

    def concat(self,
               data,
               axis: int):
        """
        Concatenate a list of BasicData objects along the specified axis

        Args:
            data(list[BasicData]): A list of BasicData objects
            axis(int): The axis along which to concatenate the TimeSeries objects
        Returns:
            BasicData

        Raise:
            ValueError

        """

        pass

    def to_numpy(self) -> np.ndarray:
        """
        Return a numpy.ndarray representation of the object

        Args:

        Returns:
            np.ndarray

        """
        return np.array(self.data)

    def to_torch(self) -> torch.tensor:

        """
        Return a torch.tensor representation of the object

        Args:
            copy(bool): Return a copy or reference.
                refer：https://pytorch.org/docs/stable/generated/torch.from_numpy.html

        Returns:
            torch.tensor

        """
        return torch.from_numpy(np.array(self.data))

    def to_dataframe(self, copy) -> pd.DataFrame:
        """
        Return a pd.DataFrame representation of the TimeSeries object

        Args:
            copy(bool):  Return a copy or reference

        Returns:
            pd.DataFrame

        """
        if copy:
            return self.data.copy()
        else:
            return self.data

    def to_taichi(self):
        """
        Return a Taichi tensor representation of the object

        Args:
            copy(bool): Return a copy or reference.
                refer：https://docs.taichi-lang.org/zh-Hans/docs/ndarray

        Returns:
            Taichi tensor

        """
        # return taichi.from_numpy(np.array(self.data))
        raise NotImplementedError

    def to_paddle(self):
        r"""
        Return a paddle tensor representation of the object
        refer：https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/tensor_en.html

        Returns:
            paddle tensor
        """
        # return paddle.to_tensor(np.array(self.data))
        raise NotImplementedError

    def to_file(self, file_path: str, file_type: str):
        """
        Save the object to a file
        Any non-abstract dataset inherited from this class should implement this method.
        Args:
            file_path(str): The path of the file to save to
            file_type(str): The type of the file to save to

        Returns:
            None
        """
        # todo: support more save type
        if file_type == "csv" or file_type == "txt":
            self.data.to_csv(file_path)
        elif file_type == "json":
            self.data.to_json(file_path)
        elif file_type == "excel":
            self.data.to_excel(file_path)
        else:
            raise ValueError("save_type only support csv/txt, json and excel!")

    def from_file(self, file_path: str, file_type: str):
        r"""
        load the object from a file
        Any non-abstract dataset inherited from this class should implement this method.
        Args:
            file_path(str): The path of the file to save to
            file_type(str): The type of the file to save to

        Returns:
            None
        """
        pass

    def indices(self) -> Sequence:
        return range(self.sample_size)



    def _select_index(self, idx: IndexType):
        r"""
            Creates a subset of the dataset from specified indices :obj:`idx`.
            Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
            list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
            long or bool.
        """
        indices = self.indices()

        if isinstance(idx, slice):
            indices = indices[idx]

        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        return indices

    def __len__(self):
        return self.sample_size

    @property
    def shape(self):
        return self.data.shape

    def visual_matplot(self, idx: int):
        """
        show one sample of the data in matplotlib
        Any non-abstract dataset inherited from this class should implement this method.
        Args:

        Returns:
            None
        """
        pass


    def _get_name(self):
        """
        get the property names of the object
        Any non-abstract dataset inherited from this class should implement this method.
        Args:
            self.name: list of str
        Returns:
            list
        """

        if self.name is None:
            return [str(i) for i in range(self.proper_size)]
        else:
            if len(self.name) != self.proper_size:
                raise ValueError("the length of name must be equal to the proper_size of the class!")
            return self.name

