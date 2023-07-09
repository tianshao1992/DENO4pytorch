#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/7/9 16:12
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : downstream_task.py
"""

import sko
import numpy as np
from Tasks.configs import _optimizer_dict, _heuristic_optimizer_dict, _gradient_optimizer_dict

class TaskOptimizer(object):
    """
        Optimizer BASIC CLASS

    """
    def __init__(self, optimal_parameters, object_function, optimizer_name=None,
                 lower_bound=None, upper_bound=None,
                 constraint_eq=tuple(), constraint_ueq=tuple(), early_stop=None,
                 **kwargs):

        self.optimizer_name = optimizer_name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.object_function = object_function
        self.optimal_parameters = optimal_parameters
        self.constraint_ueq = constraint_ueq
        self.constraint_eq = constraint_eq
        self.early_stop = early_stop

        if self.optimizer_name in _heuristic_optimizer_dict:
            self.optimizer_mode = 'heuristic'
        elif self.optimizer_name in _gradient_optimizer_dict:
            self.optimizer_mode = 'gradient'

        self._build_optimizer()

    def _build_optimizer(self, **kwargs):
        """
            build optimizer
        """

        if self.optimizer_mode == 'heuristic':
            self.basic_optimizer = _optimizer_dict[self.optimizer_name]\
                                                        (func=self.object_function,
                                                        n_dim=len(self.optimal_parameters),
                                                        lb=self.lower_bound,
                                                        ub=self.upper_bound,
                                                        constraint_eq=self.constraint_eq,
                                                        constraint_ueq=self.constraint_ueq,
                                                        **kwargs)
        elif self.optimizer_mode == 'gradient':

            NotImplementedError('gradient optimizer is not implemented yet!')

    def run(self, ):

        if self.optimizer_mode == 'heuristic':
            best_params, best_value = self.basic_optimizer.run()
        elif self.optimizer_mode == 'gradient':
            NotImplementedError('gradient optimizer is not implemented yet!')

        return best_params, best_value



