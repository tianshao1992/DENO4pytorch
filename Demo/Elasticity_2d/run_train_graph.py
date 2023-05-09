#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/27 20:34
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    :
# @File    : run_train_graph.py
"""
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from gnn.GraphNets import KernelNN3
from Utilizes.loss_metrics import FieldsLpLoss
from Utilizes.visual_data import MatplotlibVision

import sklearn.metrics
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import time
import os


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
    batch_size = dataloader.batch_size
    for batch, data in enumerate(dataloader):
        data = data.to(device)
        pred = netmodel(data)
        loss = lossfunc(pred.view(batch_size, -1), data.y.view(batch_size, -1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1) / batch_size


def valid(dataloader, netmodel, device, lossfunc):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """
    valid_loss = 0
    batch_size = dataloader.batch_size
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            data = data.to(device)
            pred = netmodel(data)
            loss = lossfunc(pred.view(batch_size, -1), data.y.view(batch_size, -1))
            valid_loss += loss.item()

    return valid_loss / (batch + 1) / batch_size


def inference(dataloader, netmodel, device):
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """

    with torch.no_grad():
        data = next(iter(dataloader))
        data = data.to(device)
        pred = netmodel(data)

    return data.x.cpu().numpy(), data.profile.cpu().numpy(), data.y.cpu().numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'Hooke-2d-Elasticity'
    work_path = os.path.join('work')
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    INPUT_PATH = './data/Meshes/Random_UnitCell_XY_10.npy'
    OUTPUT_PATH = './data/Meshes/Random_UnitCell_sigma_10.npy'

    ntrain = 1000
    nvalid = 200
    batch_size = 2
    learning_rate = 0.001

    epochs = 201
    step_size = 50
    gamma = 0.5

    # GNO
    radius = 0.2
    width = 32
    ker_width = 128
    depth = 4

    edge_features = 4
    node_features = 2
    scheduler_step = 400
    scheduler_gamma = 0.1

    ###############################################################
    # load data and data normalization
    ###############################################################
    input = np.load(INPUT_PATH)
    input = torch.tensor(input, dtype=torch.float).permute(2, 0, 1)
    # input (n, x, 2)

    output = np.load(OUTPUT_PATH)
    output = torch.tensor(output, dtype=torch.float).permute(1, 0)
    # output (n, x)

    x_train = input[:ntrain]
    y_train = output[:ntrain, None]
    x_valid = input[-nvalid:]
    y_valid = output[-nvalid:, None]

    INPUT_X = './data/Omesh/Random_UnitCell_Deform_X_10_interp.npy'
    INPUT_Y = './data/Omesh/Random_UnitCell_Deform_Y_10_interp.npy'
    inputX = np.load(INPUT_X)
    inputX = torch.tensor(inputX, dtype=torch.float).permute(2, 0, 1)
    inputY = np.load(INPUT_Y)
    inputY = torch.tensor(inputY, dtype=torch.float).permute(2, 0, 1)
    profile = torch.stack([inputX, inputY], dim=-1)[..., 0, :]
    profile_train = torch.cat((profile[:ntrain], profile[:ntrain, (0,)]), dim=1)
    profile_valid = torch.cat((profile[-nvalid:], profile[-nvalid:, (0,)]), dim=1)


    ################################################################
    # construct graphs
    ################################################################
    def get_graph_ball(mesh, radius=0.1):

        pwd = sklearn.metrics.pairwise_distances(mesh, mesh)  # (mesh_n, grid_n)
        edge_index = np.vstack(np.where(pwd <= radius))
        edge_attr = mesh[edge_index.T].reshape(-1, 4)
        return torch.tensor(edge_index, dtype=torch.long), edge_attr


    def get_graph_gaussian(mesh, sigma=0.1):
        pwd = sklearn.metrics.pairwise_distances(mesh, mesh)  # (mesh_n, grid_n)
        rbf = np.exp(-pwd ** 2 / sigma ** 2)
        sample = np.random.binomial(1, rbf)
        edge_index = np.vstack(np.where(sample))
        edge_attr = mesh[edge_index.T].reshape(-1, 4)
        return torch.tensor(edge_index, dtype=torch.long), edge_attr


    data_train = []
    for j in range(ntrain):
        edge_index, edge_attr = get_graph_gaussian(x_train[j], radius)
        data_train.append(Data(x=x_train[j], y=y_train[j], edge_index=edge_index, edge_attr=edge_attr,
                               profile=profile_train[j]))

    data_valid = []
    for j in range(nvalid):
        edge_index, edge_attr = get_graph_gaussian(x_valid[j], radius)
        data_valid.append(Data(x=x_valid[j], y=y_valid[j], edge_index=edge_index, edge_attr=edge_attr,
                               profile=profile_valid[j]))

    print(edge_index.shape, edge_attr.shape)

    ################################################################
    # Neural Networks
    ################################################################

    # 建立网络
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=False)

    ################################################################
    # Neural Networks
    ################################################################

    # 建立网络

    # Net_model = GMMNet(in_dim=2, out_dim=1, edge_dim=4, width=width, depth=depth, activation='gelu').to(device)
    Net_model = KernelNN3(32, 64, depth, 4, in_width=2, out_width=1).to(device)

    # 损失函数
    # Loss_func = nn.MSELoss()
    Loss_func = FieldsLpLoss(size_average=False)
    # L1loss = nn.SmoothL1Loss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-4)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('f',))

    star_time = time.time()
    log_loss = [[], []]

    ################################################################
    # train process
    ################################################################

    for epoch in range(epochs):

        Net_model.train()
        log_loss[0].append(train(train_loader, Net_model, device, Loss_func, Optimizer, Scheduler))

        Net_model.eval()
        log_loss[1].append(valid(valid_loader, Net_model, device, Loss_func))
        print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, learning_rate, log_loss[0][-1], log_loss[1][-1], time.time() - star_time))

        star_time = time.time()

        if epoch > 0 and epoch % 10 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(work_path, 'log_loss_graph.svg'))
            plt.close(fig)

        ################################################################
        # Visualization
        ################################################################

        if epoch > 0 and epoch % 10 == 0:
            # print('epoch: {:6d}, lr: {:.3e}, eqs_loss: {:.3e}, bcs_loss: {:.3e}, cost: {:.2f}'.
            #       format(epoch, learning_rate, log_loss[-1][0], log_loss[-1][1], time.time()-star_time))
            train_coord, train_profile, train_true, train_pred = inference(train_loader, Net_model, device)
            valid_coord, valid_profile, valid_true, valid_pred = inference(valid_loader, Net_model, device)

            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work_path, 'latest_model.pth'))

            train_coord, valid_coord = train_coord.reshape((-1, 972, 2)), valid_coord.reshape((-1, 972, 2))
            train_profile, valid_profile = train_profile.reshape((-1, 66, 2)), valid_profile.reshape((-1, 66, 2))
            train_true, valid_true = train_true.reshape((-1, 972, 1)), valid_true.reshape((-1, 972, 1))
            train_pred, valid_pred = train_pred.reshape((-1, 972, 1)), valid_pred.reshape((-1, 972, 1))

            for t in range(batch_size):
                triang = tri.Triangulation(train_coord[t][:, 0], train_coord[t][:, 1])

                fig, axs = plt.subplots(1, 3, figsize=(30, 10), num=1, layout='constrained')
                Visual.plot_fields_tr(fig, axs, train_true[t], train_pred[t], train_coord[t],
                                      triang, mask=train_profile[t].reshape((-1, 2)))
                fig.savefig(os.path.join(work_path, 'train_solution_' + str(t) + '_graph.jpg'))
                plt.close(fig)

                triang = tri.Triangulation(train_coord[t][:, 0], train_coord[t][:, 1])

                fig, axs = plt.subplots(1, 3, figsize=(30, 10), num=1, layout='constrained')
                Visual.plot_fields_tr(fig, axs, valid_true[t], valid_pred[t], valid_coord[t],
                                      triang, mask=valid_profile[t].reshape((-1, 2)))
                fig.savefig(os.path.join(work_path, 'valid_solution_' + str(t) + '_graph.jpg'))
                plt.close(fig)
