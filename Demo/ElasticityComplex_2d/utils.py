#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/5/8 2:42
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : utils.py
"""

import os
import numpy as np

def readFire(number, name, num_points):
    x_fire = []
    y_fire = []
    u_fire = []
    v_fire = []
    dTdx_fire = []
    dTdy_fire = []
    T_fire = []

    for i in range(0, number):
        file_name = os.path.join('data', name + 'x' + str(i+1) + '.txt')
        x_fire.append(np.loadtxt(file_name)[:num_points])

        file_name = os.path.join('data', name + 'y' + str(i+1) + '.txt')
        y_fire.append(np.loadtxt(file_name)[:num_points])

        file_name = os.path.join('data', name + 'u' + str(i+1) + '.txt')
        u_fire.append(np.loadtxt(file_name)[:num_points])

        file_name = os.path.join('data', name + 'v' + str(i+1) + '.txt')
        v_fire.append(np.loadtxt(file_name)[:num_points])

        file_name = os.path.join('data', name + 'dTdx' + str(i+1) + '.txt')
        dTdx_fire.append(np.loadtxt(file_name)[:num_points])

        file_name = os.path.join('data', name + 'dTdy' + str(i+1) + '.txt')
        dTdy_fire.append(np.loadtxt(file_name)[:num_points])

        file_name = os.path.join('data', name + 'T' + str(i+1) + '.txt')
        T_fire.append(np.loadtxt(file_name)[:num_points])


    x_fire = np.stack(x_fire, axis=0)
    y_fire = np.stack(y_fire, axis=0)
    u_fire = np.stack(u_fire, axis=0)
    v_fire = np.stack(v_fire, axis=0)
    dTdx_fire = np.stack(dTdx_fire, axis=0)
    dTdy_fire = np.stack(dTdy_fire, axis=0)
    T_fire = np.stack(T_fire, axis=0)


    all_data = np.stack((x_fire, y_fire, u_fire, v_fire, dTdx_fire, dTdy_fire, T_fire), axis=-1)

    return all_data


if __name__ == "__main__":
    square_data = readFire(90, 'square', num_points=1200)