#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/5/9 16:05
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : geometrics.py
"""

import numpy as np

def ccw_sort(points):
    """Sort given polygon points in CCW order"""
    points = np.array(points)
    mean = np.mean(points, axis=0)
    coords = points - mean
    s = np.arctan2(coords[:, 0], coords[:, 1])
    return points[np.argsort(s), :]
