# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/6/6 11:19
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：run_FNO.py.py
@File ：run_FNO.py.py
"""
import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
import yaml
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary

# add .py path
file_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(file_path.split('Demo')[0]))

from Models.cnn.ConvNets import UNet3d
from Models.fno.FNOs import FNO3d
from Models.transformer.Transformers import FourierTransformer
from Utilizes.loss_metrics import FieldsLpLoss
from Utilizes.parallel import setup_DDP
from Utilizes.process_data import DataNormer
from Utilizes.visual_data import MatplotlibVision, TextLogger


def reduce_mean(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM) # sum-up as the all-reduce operation
    rt /= world_size # NOTE this is necessary, since all_reduce here do not perform average 
    return rt


def feature_transform(x):
    """
    Args:
        x: input coordinates
    Returns:
        res: input transform
    """
    shape = x.shape
    batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
    gridx = torch.linspace(0, 1, size_x, dtype=torch.float32)
    gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    gridy = torch.linspace(0, 1, size_y, dtype=torch.float32)
    gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    gridz = torch.linspace(0, 1, size_z, dtype=torch.float32)
    gridz = gridz.reshape(1, 1, 1, size_y, 1).repeat([batchsize, size_x, size_y, 1, 1])
    return torch.cat((gridx, gridy, gridz), dim=-1).to(x.device)


def train(dataloader, netmodel, local_rank, lossfunc, lossmetric, optimizer, scheduler):
    """
    Args:
        data_loader: output fields at last time step
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """
    train_loss = 0
    total_size = 0
    train_metric = 0
    for batch, (xx, yy) in enumerate(dataloader):
        input_sizes = list(xx.shape)
        xx = xx.reshape(input_sizes[:-2] + [-1, ])
        xx = xx.cuda(local_rank, non_blocking=True)
        yy = yy.cuda(local_rank, non_blocking=True)
        gd = feature_transform(xx)

        pred = netmodel(xx, gd)
        loss = lossfunc(pred, yy)
        metric = lossmetric(pred, yy).mean()

        torch.distributed.barrier() # important, TODO [同步点]

        reduced_loss = reduce_mean(loss, world_size) # 这里调用的是all_reduce，来归一不同gpu上的结果
        reduced_metric = reduce_mean(metric, world_size)
        
        train_loss += reduced_loss.item() * input_sizes[0]
        train_metric += reduced_metric.item() * input_sizes[0]
        total_size += input_sizes[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    return train_loss / total_size, train_metric / total_size


def valid(dataloader, netmodel, local_rank, lossfunc, lossmetric):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """
    valid_loss = 0
    total_size = 0
    valid_metric = 0
    with torch.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            input_sizes = list(xx.shape)
            xx = xx.reshape(input_sizes[:-2] + [-1, ])
            xx = xx.cuda(local_rank, non_blocking=True)
            yy = yy.cuda(local_rank, non_blocking=True)
            gd = feature_transform(xx)

            pred = netmodel(xx, gd)
            loss = lossfunc(pred, yy)
            metric = lossmetric(pred, yy).mean()

            torch.distributed.barrier() # important, TODO [同步点]

            reduced_loss = reduce_mean(loss, world_size) # 这里调用的是all_reduce，来归一不同gpu上的结果
            reduced_metric = reduce_mean(metric, world_size)
            
            valid_loss += reduced_loss.item() * input_sizes[0]
            valid_metric += reduced_metric.item() * input_sizes[0]
            total_size += input_sizes[0]

        return valid_loss / total_size, valid_metric / total_size


class custom_dataset(Dataset):
    def __init__(self, data, input_step):
        self.data = data
        self.input_step = input_step
        self.sample_size = data.shape[0]
        self.step_size = data.shape[1]

    def __getitem__(self, idx):  # 根据 idx 取出其中一个

        idt, ids = divmod(idx, self.sample_size)
        x = self.data[ids, idt:idt + self.input_step]
        y = self.data[ids, idt + self.input_step]
        return x.transpose((1, 2, 3, 4, 0)), y

    def __len__(self):  # 总数据的多少
        return self.sample_size * (self.step_size - self.input_step)



if __name__ == "__main__":
    ################################################################
    # config
    ################################################################

    rank, local_rank, world_size, device = setup_DDP()

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)    # 该变量由torch框架自动设置，并且设置为-1时表示不分配
    parser.add_argument("--model_name", default='FNO', type=str) # 模型名称
    parser.add_argument("--batch_size", default=8, type=int)     # 每块显卡的batch大小
    parser.add_argument("--weight_decay", default=1e-8, type=float)     # L2正则权重
    parser.add_argument("--total_epoch", default=101, type=int)     # 总训练epoch数
    parser.add_argument("--learning_rate", default=0.001, type=float)     # 学习率
    args = parser.parse_args()

    model_name = args.model_name
    work_path = os.path.join('work', model_name)
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    # 将控制台的结果输出到log文件
    Logger = TextLogger(os.path.join(work_path, 'train.log'))
    if rank == 0:
        Logger.info("model_name: {:s}, computing device: {:s}".format(model_name, str(device)))

    in_dim = 3
    out_dim = 3
    steps = 5

    # 空间分辨率
    s = 64
    r1 = 1
    r2 = 1
    r3 = 1
    s1 = int(((s - 1) / r1) + 1)
    s2 = int(((s - 1) / r2) + 1)
    s3 = int(((s - 1) / r3) + 1)

    batch_size = args.batch_size
    total_epoch = args.total_epoch
    learning_rate = args.learning_rate
    scheduler_step = int(total_epoch * 0.8)
    scheduler_gamma = 0.1

    if rank == 0:
        Logger.info('total_epoch: {:d}, learning_rate: {:e}, scheduler_step: {:d}, scheduler_gamma: {:e}'
                    .format(total_epoch, learning_rate, scheduler_step, scheduler_gamma))


    ################################################################
    # load data
    ################################################################

    
    train_data_name = '../DENO4pytorch/Demo/Turbulence_3d+t/data/vel_{:d}-{:d}g_600p_gap200_LES64.npy'.format(local_rank*20+1, (local_rank+1)*20)
    train_data = np.load(train_data_name)  #默认 每个卡分配部分数据集

    
    valid_data_name = '../DENO4pytorch/Demo/Turbulence_3d+t/data/vel_{:d}-{:d}g_600p_gap200_LES64.npy'.format(181, 200)
    valid_data = np.load(valid_data_name)  #默认 181-200 为测试集
    valid_data = valid_data[-10:]

    train_dataset = custom_dataset(train_data, input_step=steps)
    valid_dataset = custom_dataset(valid_data, input_step=steps)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)    # NOTE, 分布式mini-batch数据采样
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=2, pin_memory=True,
                                               batch_size=batch_size, drop_last=True, sampler=train_sampler)

    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=True)    # NOTE, 分布式mini-batch数据采样
    valid_loader = torch.utils.data.DataLoader(valid_dataset, num_workers=2, pin_memory=True,
                                               batch_size=batch_size, drop_last=True, sampler=valid_sampler)
    


    Logger.info("local_rank: {:d}, train_data_name: {:s}, train data sizes: {}".format(local_rank, train_data_name, train_data.shape))

    ################################################################
    # Neural Networks
    ################################################################

    # 建立网络

    if 'FNO' in model_name:

        mode = 16
        modes = (mode, mode, mode)
        width = 64
        depth = 4
        
        padding = 0
        dropout = 0.0
        Net_model = FNO3d(in_dim=in_dim, out_dim=out_dim, modes=modes, width=width, depth=depth, steps=steps,
                          padding=padding, activation='gelu', use_complex=False).cuda(local_rank)  # 把模型推送到当前线程所在的gpu的内存

    if 'CNN' in model_name:
        width = 64
        depth = 4
        padding = 0
        dropout = 0.0
        Net_model = UNet3d(in_sizes=(s, s, s, in_dim), out_sizes=(s, s, s, out_dim), width=width, depth=depth, steps=steps,
                           activation='gelu').cuda(local_rank)       # 把模型推送到当前线程所在的gpu的内存

    if 'Trans' in model_name:
        
        with open(os.path.join('./Demo/Turbulence_3d+t/transformer_config.yml')) as f:
            config = yaml.full_load(f)

        config = config['Turbulence_3d+t']
        Net_model = FourierTransformer(**config).cuda(local_rank)       # 把模型推送到当前线程所在的gpu的内存

        with open(os.path.join(work_path, 'model_config.yml'), 'w') as file:
            file.write(yaml.dump({'Turbulence_3d+t': Net_model.config}, allow_unicode=True))

    Net_model = torch.nn.parallel.DistributedDataParallel(Net_model, device_ids=[local_rank])

    if rank == 0:
        # Logger.info("Total Number of Parameters: {:d}".format(Net_model.parameters()[0].numel()))
        input1 = torch.randn(batch_size, s, s, s, steps * in_dim).cuda(local_rank, non_blocking=True)
        input2 = torch.randn(batch_size, s, s, s, in_dim).cuda(local_rank, non_blocking=True)
        model_statistics = summary(Net_model, input_data=[input1, input2], device=device, verbose=0)
        Logger.write(str(model_statistics))


    # 损失函数
    Loss_func = nn.MSELoss()
    Loss_metirc = FieldsLpLoss(d=2, p=2, reduction=True, size_average=False)
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=args.weight_decay)
    cudnn.benchmark = True
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y', 'z'), field_name=('u', 'v', 'w'))

    star_time = time.time()
    log_loss = [[], []]

    ################################################################
    # train process
    ################################################################

    for epoch in range(total_epoch):

        Net_model.train()
        log_loss[0].append(train(train_loader, Net_model, local_rank, Loss_func, Loss_metirc, Optimizer, Scheduler))

        Net_model.eval()
        log_loss[1].append(valid(valid_loader, Net_model, local_rank, Loss_func, Loss_metirc))

        if rank == 0:
            Logger.info('model_name: {:s}, epoch: {:5d}, lr: {:.3e}, '
                        'train_step_loss: {:.3e}, valid_step_loss: {:.3e}, '
                        'train_step_L2error: {:.3f}, valid_step_L2error: {:.3f}, '
                        'cost: {:.2f}'.
                        format(model_name, epoch, Optimizer.param_groups[0]['lr'],
                            log_loss[0][-1][0], log_loss[1][-1][0],
                            log_loss[0][-1][1], log_loss[1][-1][1],
                            time.time() - star_time))

        star_time = time.time()

        if rank == 0 and epoch > 0 and epoch % 1 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :, 0], 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :, 0], 'valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(work_path, 'log_loss.svg'))
            plt.close(fig)

            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=2)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :, 1], 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :, 1], 'valid_step')
            fig.suptitle('training metrics')
            fig.savefig(os.path.join(work_path, 'log_metric.svg'))
            plt.close(fig)


        if rank == 0 and epoch > 0 and epoch % 20 == 0:

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
            os.path.join(work_path, 'latest_model.pth'))
