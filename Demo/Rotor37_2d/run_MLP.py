# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/4/17 22:06
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：run_FNO.py
@File ：run_FNO.py
"""
import numpy as np
import torch
import torch.nn as nn
from Utilizes.visual_data import MatplotlibVision
from Utilizes.process_data import DataNormer, MatLoader
from collections import OrderedDict

import matplotlib.pyplot as plt
import time
import os
from utilizes_rotor37 import get_grid, get_origin
from post_process.post_data import Post_2d

class MLP(nn.Module):
    def __init__(self, layers=None, is_BatchNorm=True,
                 in_dim=None,
                 out_dim=None,
                 n_hidden=None,
                 num_layers=None):
        if layers is None:
            layers = [in_dim]
            for ii in range(num_layers-2):
                layers.append(n_hidden)
            layers.append(out_dim)
        super(MLP, self).__init__()
        self.depth = len(layers)
        self.activation = nn.GELU
        #先写完整的layerslist
        layer_list = []
        for i in range(self.depth-2):
            layer_list.append(('layer_%d' % i, nn.Linear(layers[i], layers[i+1])))
            if is_BatchNorm is True:
                layer_list.append(('batchnorm_%d' % i, nn.BatchNorm1d(layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))

        #最后一层，输出层
        layer_list.append(('layer_%d' % (self.depth-2), nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)
        #再直接使用sequential生成网络
        self.layers = nn.Sequential(layerDict)

    def forward(self,x):
        y = self.layers(x)
        return y

def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    """
    Args:
        data_loader: output fields at last time step
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """
    train_loss = 0
    for batch, (input,output) in enumerate(dataloader):
        input = input.to(device)
        output = output.to(device)
        pred = netmodel(input)

        loss = lossfunc(pred, output)

        # grid_size = 64
        # weighted_lines = 3
        # weighted_cof = 1.5
        # temp1 = np.ones([weighted_lines, grid_size])*weighted_cof
        # temp2 = np.ones([grid_size-weighted_lines*2, grid_size])
        # weighted_mat = np.concatenate((temp1,temp2,temp1),axis=0).reshape(output[0].shape)
        # weighted_mat = np.tile(weighted_mat[None,:],(output.shape[0],1))

        optimizer.zero_grad()
        loss.backward() # 自动微分
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1)


def valid(dataloader, netmodel, device, lossfunc):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """
    valid_loss = 0
    with torch.no_grad():
        for batch, (input, output) in enumerate(dataloader):
            input = input.to(device)
            output = output.to(device)
            pred = netmodel(input)

            loss = lossfunc(pred, output)
            valid_loss += loss.item()

    return valid_loss / (batch + 1)

def inference(dataloader, netmodel, device):
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """
    with torch.no_grad():
        xx, yy = next(iter(dataloader))
        xx = xx.to(device)
        pred = netmodel(xx)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), yy.numpy(), pred.cpu().numpy()


if __name__ == "__main__":
################################################################
# configs
################################################################
    name = 'MLP'
    work_path = os.path.join('work', name)
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    in_dim = 28
    out_dim = 5

    ntrain = 2700
    nvalid = 250

    batch_size = 32
    epochs = 1001

    learning_rate = 0.001
    scheduler_step = 800
    scheduler_gamma = 0.1

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

################################################################
# load data
################################################################

    design, fields = get_origin() #获取原始数据

    input = design
    input = torch.tensor(input, dtype=torch.float)
    # output = fields[:, 0, :, :, :].transpose((0, 2, 3, 1))
    output = fields
    # output = output.reshape([output.shape[0],-1])
    output = torch.tensor(output, dtype=torch.float)
    print(input.shape, output.shape)

    train_x = input[:ntrain, :]
    train_y = output[:ntrain, :]
    valid_x = input[-nvalid:, :]
    valid_y = output[-nvalid:, :]

    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    x_normalizer.load(os.path.join(work_path,'x_norm.pkl'))
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)

    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    y_normalizer.load(os.path.join(work_path,'y_norm.pkl'))
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)

    train_y = train_y.reshape([train_x.shape[0],-1])
    valid_y = valid_y.reshape([valid_x.shape[0],-1])

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                               batch_size=batch_size, shuffle=False, drop_last=True)


################################################################
# Neural Networks
################################################################

    # 建立网络
    layer_mat = [in_dim, 256, 256, 256, 256, 256, 256, 256, 256, out_dim*64*64]
    Net_model =  MLP(layer_mat=layer_mat, is_BatchNorm=False)
    Net_model = Net_model.to(Device)
    print(name)
    # summary(Net_model, input_size=(batch_size, train_x.shape[1]), device=Device)

    # 损失函数
    Loss_func = nn.MSELoss()
    # Loss_func = nn.SmoothL1Loss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-4)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('p', 't', 'rho', 'alf', 'v'))

    star_time = time.time()
    log_loss = [[], []]

    ################################################################
    # train process
    ################################################################
    grid = get_grid()

    for epoch in range(epochs):

        Net_model.train()
        log_loss[0].append(train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))

        Net_model.eval()
        log_loss[1].append(valid(valid_loader, Net_model, Device, Loss_func))
        print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, Optimizer.param_groups[0]['lr'], log_loss[0][-1], log_loss[1][-1], time.time() - star_time))

        star_time = time.time()

    ################################################################
    # Visualization
    ################################################################

        if epoch > 0 and epoch % 5 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(work_path, 'log_loss.svg'))
            plt.close(fig)



        if epoch > 0 and epoch % 100 == 0:
            train_coord, train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_coord, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))

            train_true = train_true.reshape([train_true.shape[0], 64, 64, out_dim])
            train_pred = train_pred.reshape([train_pred.shape[0], 64, 64, out_dim])
            valid_true = valid_true.reshape([valid_true.shape[0], 64, 64, out_dim])
            valid_pred = valid_pred.reshape([valid_pred.shape[0], 64, 64, out_dim])

            for fig_id in range(5):
                fig, axs = plt.subplots(out_dim, 3, figsize=(18, 20), num=2)

                Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id],grid)
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(5):
                fig, axs = plt.subplots(out_dim, 3, figsize=(20, 15), num=3)
                Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id],grid)
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            train_true = train_true.reshape([train_true.shape[0], 64, 64, out_dim])
            train_pred = train_pred.reshape([train_pred.shape[0], 64, 64, out_dim])
            valid_true = valid_true.reshape([valid_true.shape[0], 64, 64, out_dim])
            valid_pred = valid_pred.reshape([valid_pred.shape[0], 64, 64, out_dim])

            train_true = y_normalizer.back(train_true)
            train_pred = y_normalizer.back(train_pred)
            valid_true = y_normalizer.back(valid_true)
            valid_pred = y_normalizer.back(valid_pred)

            for fig_id in range(5):
                post_true = Post_2d(train_true[fig_id], grid)
                post_pred = Post_2d(train_pred[fig_id], grid)
                # plt.plot(post_true.Efficiency[:,-1],np.arange(64),label="true")
                # plt.plot(post_pred.Efficiency[:, -1], np.arange(64), label="pred")
                fig, axs = plt.subplots(1, 1, figsize=(10, 5), num=1)
                Visual.plot_value(fig, axs, post_true.Efficiency[:, -1], np.arange(64), label="true")
                Visual.plot_value(fig, axs, post_pred.Efficiency[:, -1], np.arange(64), label="pred",
                                  title="train_solution", xylabels=("efficiency", "span"))
                fig.savefig(os.path.join(work_path, 'train_solution_eff_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(5):
                post_true = Post_2d(valid_true[fig_id], grid)
                post_pred = Post_2d(valid_pred[fig_id], grid)
                fig, axs = plt.subplots(1, 1, figsize=(10, 5), num=1)
                Visual.plot_value(fig, axs, post_true.Efficiency[:, -1], np.arange(64), label="true")
                Visual.plot_value(fig, axs, post_pred.Efficiency[:, -1], np.arange(64), label="pred",
                                  title="train_solution", xylabels=("efficiency", "span"))
                fig.savefig(os.path.join(work_path, 'valid_solution_eff_' + str(fig_id) + '.jpg'))
                plt.close(fig)
