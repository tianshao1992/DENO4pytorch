#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/5/8 2:42
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : utils.py
"""


def readFire(number, name):
    coord = 0
    with open('/scratch/users/kashefi/PIPNSolid/data20/' + name + 'x' + str(number) + '.txt', 'r') as f:
        for line in f:
            x_fire[coord] = float(line.split()[0])
            coord += 1
    f.close()

    coord = 0
    with open('/scratch/users/kashefi/PIPNSolid/data20/' + name + 'y' + str(number) + '.txt', 'r') as f:
        for line in f:
            y_fire[coord] = float(line.split()[0])
            coord += 1
    f.close()

    coord = 0
    with open('/scratch/users/kashefi/PIPNSolid/data20/' + name + 'u' + str(number) + '.txt', 'r') as f:
        for line in f:
            u_fire[coord] = float(line.split()[0]) * 1
            coord += 1
    f.close()

    coord = 0
    with open('/scratch/users/kashefi/PIPNSolid/data20/' + name + 'v' + str(number) + '.txt', 'r') as f:
        for line in f:
            v_fire[coord] = float(line.split()[0]) * 1
            coord += 1
    f.close()

    coord = 0
    with open('/scratch/users/kashefi/PIPNSolid/data20/' + name + 'dTdx' + str(number) + '.txt', 'r') as f:
        for line in f:
            dTdx_fire[coord] = float(line.split()[0]) / 1
            coord += 1
    f.close()

    coord = 0
    with open('/scratch/users/kashefi/PIPNSolid/data20/' + name + 'dTdy' + str(number) + '.txt', 'r') as f:
        for line in f:
            dTdy_fire[coord] = float(line.split()[0]) / 1
            coord += 1
    f.close()

    coord = 0
    with open('/scratch/users/kashefi/PIPNSolid/data20/' + name + 'T' + str(number) + '.txt', 'r') as f:
        for line in f:
            T_fire[coord] = float(line.split()[0]) / 1
            coord += 1
    f.close()
