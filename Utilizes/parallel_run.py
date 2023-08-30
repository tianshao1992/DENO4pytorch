#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/8/30 18:45
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : parallel_run.py
# @Description    : ******
"""

import os
import sys
import time
import torch
import torch.distributed as dist


def setup_DDP(backend="nccl", verbose=False):
    """
    We don't set ADDR and PORT in here, like:
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
    Because program's ADDR and PORT can be given automatically at startup.
    E.g. You can set ADDR and PORT by using:
        python -m torch.distributed.launch --master_addr="192.168.1.201" --master_port=23456 ...

    You don't set rank and world_size in dist.init_process_group() explicitly.

    :param backend:
    :param verbose:
    :return:
    """
    rank = int(os.environ["RANK"])  # get rank
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # If the OS is Windows or macOS, use gloo instead of nccl
    dist.init_process_group(backend=backend)  # NOTE, init 线程组(process group) using nccl, gloo
    # set distributed device
    device = torch.device("cuda:{}".format(local_rank))
    torch.cuda.set_device(local_rank)  # NOTE, local_rank是当前的一个gpu，其可以是0, 1, 2, 3 (当有4个gpu的时候）

    if verbose:
        print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}, compute device: {device}")

    return rank, local_rank, world_size, device


if __name__ == "__main__":
    # setup_DDP()
    # print("rank: ", dist.get_rank())
    # print("local_rank: ", dist.get_local_rank())
    # print("world_size: ", dist.get_world_size())
    # print("device: ", dist.get_device())
    rank, local_rank, world_size, device = setup_DDP(verbose=True)
