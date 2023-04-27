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
from torch.utils.data import DataLoader
from torchinfo import summary
from fno.FNOs import FNO2d
from cnn.ConvNets import UNet2d
from Utilizes.visual_data import MatplotlibVision
from Utilizes.process_data import DataNormer
from collections import OrderedDict

import matplotlib.pyplot as plt
import time
import os

class MLP(nn.Module):
    def __init__(self, layers, is_BatchNorm=True):
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


def get_grid():
    xx = np.linspace(-0.127, 0.126, 64)
    xx = np.tile(xx, [64,1])

    hub_file = os.path.join('data', 'hub_lower.txt')
    hub = np.loadtxt(hub_file)
    shroud_files = os.path.join('data', 'shroud_upper.txt')
    shroud = np.loadtxt(shroud_files)

    yy = []
    for i in range(64):
        yy.append(np.linspace(hub[i],shroud[i],64))

    yy = np.concatenate(yy, axis=0)
    yy = yy.reshape(64, 64).T
    xx = xx.reshape(64, 64)

    return np.concatenate([xx[:,:,np.newaxis],yy[:,:,np.newaxis]],axis=2)



def get_origin():
    # sample_num = 500
    # sample_start = 0

    design_files = [os.path.join('data', 'rotor37_600_sam.dat'),
                    os.path.join('data', 'rotor37_900_sam.dat')]
    field_paths = [os.path.join('data', 'Rotor37_span_600_data_64cut_clean'),
                   os.path.join('data', 'Rotor37_span_900_data_64cut_clean')]

    fields = []
    case_index = []
    for path in field_paths:
        names = os.listdir(path)
        fields.append([])
        case_index.append([])
        for i in range(len(names)):
            # 处理后数据格式为<class 'tuple'>: (3, 5, 64，64)
            if 'case_' + str(i) + '.npy' in names:
                fields[-1].append(np.load(os.path.join(path, 'case_' + str(i) + '.npy'))
                                  .astype(np.float32).transpose((1, 2, 0, 3)))
                case_index[-1].append(i)
        fields[-1] = np.stack(fields[-1], axis=0)

    design = []
    for i, file in enumerate(design_files):
        design.append(np.loadtxt(file, dtype=np.float32)[case_index[i]])

    design = np.concatenate(design, axis=0)
    fields = np.concatenate(fields, axis=0)

    return design, fields


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

        grid_size = 64

        weighted_lines = 3
        weighted_cof = 1.5
        temp1 = np.ones([weighted_lines, grid_size])*weighted_cof
        temp2 = np.ones([grid_size-weighted_lines*2, grid_size])
        weighted_mat = np.concatenate((temp1,temp2,temp1),axis=0).reshape(output[0].shape)
        weighted_mat = np.tile(weighted_mat[None,:],(output.shape[0],1))

        optimizer.zero_grad()
        loss.backward()
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
    # grid = get_grid()


    name = 'MLP'
    work_path = os.path.join('work', name)
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    design, fields = get_origin() #获取原始数据

    in_dim = 28
    out_dim = 5

    ntrain = 800
    nvalid = 300

    batch_size = 32
    epochs = 1000
    learning_rate = 0.001
    scheduler_step = 800
    scheduler_gamma = 0.1

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

################################################################
# load data
################################################################

    input = design
    input = torch.tensor(input, dtype=torch.float)

    output = fields[:, 0, :, :, :].transpose((0, 2, 3, 1))
    # output = output.reshape([output.shape[0],-1])
    output = torch.tensor(output, dtype=torch.float)
    print(input.shape, output.shape)

    train_x = input[:ntrain, :]
    train_y = output[:ntrain, :]
    valid_x = input[ntrain:ntrain + nvalid, :]
    valid_y = output[ntrain:ntrain + nvalid, :]

    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)

    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
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
    Net_model =  MLP(layer_mat,is_BatchNorm=False)
    Net_model = Net_model.to(Device)
    print(name)
    summary(Net_model, input_size=(batch_size, train_x.shape[1]), device=Device)

    # 损失函数
    Loss_func = nn.MSELoss()
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


        if epoch > 0 and epoch % 50 == 0:
            train_coord, train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_coord, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))

            train_true = train_true.reshape([train_true.shape[0], 64, 64, out_dim])
            train_pred = train_pred.reshape([train_pred.shape[0], 64, 64, out_dim])
            valid_true = valid_true.reshape([valid_true.shape[0], 64, 64, out_dim])
            valid_pred = valid_pred.reshape([valid_pred.shape[0], 64, 64, out_dim])

            for fig_id in range(10):
                fig, axs = plt.subplots(out_dim, 3, figsize=(18, 20), num=2)
                Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id],grid)
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(10):
                fig, axs = plt.subplots(out_dim, 3, figsize=(18, 20), num=3)
                Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id],grid)
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)
