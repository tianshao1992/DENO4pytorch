# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/6/20 1:15
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：infer_Trans.py
@File ：infer_Trans.py
"""
import argparse
import os
import sys
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
import matplotlib as mpl
import matplotlib.pyplot as plt

# add .py path
file_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(file_path)
sys.path.append(os.path.join(file_path.split('Demo')[0]))

from Models.cnn.ConvNets import UNet3d
from Models.fno.FNOs import FNO3d
from Models.transformer.Transformers import FourierTransformer
from Utilizes.visual_data import MatplotlibVision, TextLogger
from Utilizes.process_data import DataNormer
from Utilizes.loss_metrics import FieldsLpLoss
from run_train_DDP import feature_transform, custom_dataset


def calc_spectra3d(var,kk,N):
    nek = N//3
    var_k = np.fft.fftn(var, axes=(0,1,2))/(1.0*(N**3))
    e_k = 0.5*np.real(var_k[:,:,:,0]*np.conj(var_k[:,:,:,0])+var_k[:,:,:,1]*np.conj(var_k[:,:,:,1])+var_k[:,:,:,2]*np.conj(var_k[:,:,:,2]))
    k = np.arange(1,nek)
    Ek = np.zeros(shape=(nek-1,), dtype=np.float64)
    cond = (np.sqrt(kk)+0.5).astype(int)
    for i in range(1,nek):
        Ek[i-1] = np.sum(e_k[cond == i], dtype=np.float64)
    return np.stack((k, Ek), axis=-1)

if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", default=0, type=int)    # 该变量由torch框架自动设置，并且设置为-1时表示不分配
    parser.add_argument("--model_name", default='FNO', type=str) # 模型名称
    parser.add_argument('--total_infer_steps', type=int, default=300)  # 连续推理步数
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    model_name = args.model_name
    work_path = os.path.join('work', model_name)
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)
    isCreated = os.path.exists(os.path.join(work_path, 'infer'))
    if not isCreated:
        os.makedirs(os.path.join(work_path, 'infer'))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # 将控制台的结果输出到log文件
    Logger = TextLogger(os.path.join(work_path, 'infer.log'))
    Logger.info("model_name: {:s}, computing device: {:s}".format(model_name, str(device)))

    in_dim = 3
    out_dim = 3
    steps = 5
    total_infer_steps = args.total_infer_steps


    ################################################################
    # load data
    ################################################################

    # infer_data_name = '../DENO4pytorch/Demo/Turbulence_3d+t/data/vel_{:d}-{:d}g_600p_gap200_LES64.npy'.format(181, 200)
    # infer_data_name = '../DENO4pytorch/Demo/Turbulence_3d+t/data/HIT_vel_50g_600p_gap200_32.npy'
    infer_data = np.load(infer_data_name)  #默认 181-200 为测试集
    total_infer_cases = infer_data.shape[0]

    s = infer_data.shape[2]
    r1 = 1
    r2 = 1
    r3 = 1
    s1 = int(((s - 1) / r1) + 1)
    s2 = int(((s - 1) / r2) + 1)
    s3 = int(((s - 1) / r3) + 1)

    ################################################################
    # Neural Networks
    ################################################################

    if 'FNO' in model_name:

        mode = 16
        modes = (mode, mode, mode)
        width = 64
        depth = 4
        padding = 0
        dropout = 0.0
        Net_model = FNO3d(in_dim=in_dim, out_dim=out_dim, modes=modes, width=width, depth=depth, steps=steps,
                          padding=padding, activation='gelu', use_complex=False).to(device)  # 把模型推送到当前线程所在的gpu的内存

    if 'CNN' in model_name:
        width = 64
        depth = 4
        padding = 0
        dropout = 0.0
        Net_model = UNet3d(in_sizes=(s, s, s, in_dim), out_sizes=(s, s, s, out_dim), width=width, depth=depth, steps=steps,
                           activation='gelu').to(device)       # 把模型推送到当前线程所在的gpu的内存

    if 'Trans' in model_name:
        
        with open(os.path.join(work_path, 'model_config.yml')) as f:
            config = yaml.full_load(f)

        config = config['Turbulence_3d+t']
        Net_model = FourierTransformer(**config).to(device)       # 把模型推送到当前线程所在的gpu的内存

    # 模型载入
    # try:
    checkpoint = torch.load(os.path.join(work_path, 'latest_model.pth'))
    Logger.warning("model path: {:s}".format(os.path.join(work_path, 'latest_model.pth')))

    weights_dict = {}
    for k, v in checkpoint['net_model'].items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
    
    Net_model.load_state_dict(weights_dict)

    Logger.warning("model load successful!")
    # except:
        # Logger.warning("model doesn't exist!")


    xx = torch.randn(total_infer_cases, s, s, s, in_dim, steps).to(device)
    yy = torch.randn(total_infer_cases, s, s, s, out_dim).to(device)
    input_sizes = list(xx.shape)
    xx = xx.reshape(input_sizes[:-2] + [-1, ])
    xx = xx.to(device)
    yy = yy.to(device)
    grid = feature_transform(xx)
    model_statistics = summary(Net_model, input_data=[xx, grid], device=device, verbose=0)
    Logger.write(str(model_statistics))

    # 损失函数
    Loss_metirc = FieldsLpLoss(d=2, p=2, reduction=True, size_average=False)
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('u', 'v', 'w'))

    log_loss = [[], []]
    log_per = [[], []]

    ################################################################
    # infer process
    ################################################################

    initial_input = infer_data[:, :steps]

    preds = []
    truth = infer_data[:, steps:total_infer_steps + steps]
    Net_model.eval() # 切换模型为评估模式
    with torch.no_grad():
        for t in range(total_infer_steps):
            star_time = time.time()
            if t == 0:
                x = torch.tensor(initial_input.transpose((0, 2, 3, 4, 5, 1)), dtype=torch.float32).to(device)
                input_sizes = list(x.shape)
            else:
                x = torch.cat((x[..., -steps + 1:], yy), dim=-1)

            xx = x.reshape(input_sizes[:-2] + [-1, ])
            grid = feature_transform(xx)
            yy = Net_model(xx, grid).unsqueeze(-1)
            preds.append(yy.cpu().numpy())

            Logger.info("time step: {:d}, cost: {:.2f}".format(t, time.time() - star_time))

    preds = np.concatenate(preds, axis=-1).transpose((0, 5, 1, 2, 3, 4))
    coord = grid.cpu().numpy()
    L2_error = []
    for t in range(total_infer_steps):
        L2_error.append(Loss_metirc(preds[:, t].reshape([input_sizes[0], -1, 1]),
                                 truth[:, t].reshape([input_sizes[0], -1, 1])))

    L2_error = np.stack(L2_error, axis=1)
    print(L2_error.shape)
    avg_error = np.mean(L2_error, axis=0)
    std_error = np.std(L2_error, axis=0)

    np.savetxt(os.path.join(work_path, '{}.txt'.format('time_lploss')), np.concatenate((avg_error, std_error), axis=-1))

    fig, axs = plt.subplots(1, 1, figsize=(10, 5), num=2, constrained_layout=True)
    Visual.plot_value(fig, axs, 1+np.arange(total_infer_steps), avg_error, std=std_error, label='relative_l2_loss',
                          xylabels=('time_steps', 'lp_loss'))

    fig.savefig(os.path.join(work_path, '{}.svg'.format('time_lploss')))
    plt.close(fig)

    truth_spectrum = []
    preds_spectrum = []

    # 流场输出  
    for case_id in range(total_infer_cases):

        #######################################转换到谱空间
        Nx,Ny,Nz,nvar = np.shape(truth[case_id, 0,:,:,:,:])
        kx = np.fft.fftfreq(Nx,1.0/Nx).astype(int)
        ky = np.fft.fftfreq(Ny,1.0/Ny).astype(int)
        kz = np.fft.fftfreq(Nz,1.0/Nz).astype(int)
        kkx, kky, kkz = np.meshgrid(kx,ky,kz,sparse=False, indexing="ij")
        kk = np.square(kkx)+np.square(kky)+np.square(kkz)
        # print("kx:", kx)
        # print("ky:", ky)
        # print("kz:", kz)
        # print("kk:", kk)

        for time_id in range(0, total_infer_steps, 50):
            fig, axs = plt.subplots(3, 3, figsize=(25, 25), num=1, layout='constrained')
            Visual.plot_fields_ms(fig, axs, truth[case_id, time_id, ..., s//2, :],
                                  preds[case_id, time_id, ..., s//2, :], coord[case_id, ..., s//2, :2],
                                  titles=['fDNS', model_name, 'error'])
            title = 'case_{:d}_step_{:d}: l2error: {:.3e}'.format(case_id, time_id, float(L2_error[case_id, time_id]))
            print(title)
            fig.suptitle(title)
            fig.savefig(os.path.join(work_path, 'infer', 'velocity_case_{}_step_{}.jpg'.format(case_id, time_id)))
            plt.close(fig)

            truth_spectrum.append(calc_spectra3d(truth[case_id, time_id], kk, Nx))
            preds_spectrum.append(calc_spectra3d(preds[case_id, time_id], kk, Nx))

            np.savetxt(os.path.join(work_path, 'infer','spectrum_case_{}_step_{}.txt'.format(case_id, time_id)),
                       np.concatenate((truth_spectrum[-1], preds_spectrum[-1]), axis=-1))

            fig, axs = plt.subplots(1, 1, figsize=(8, 8), num=2, layout='constrained')
            Visual.plot_value(fig, axs, truth_spectrum[-1][:, 0], truth_spectrum[-1][:, 1], label='fDNS',
                              axis_log=(True, True), xylabels=('k', 'E(k)'))
            Visual.plot_value(fig, axs, preds_spectrum[-1][:, 0], preds_spectrum[-1][:, 1], label=model_name, 
                              axis_log=(True, True), xylabels=('k', 'E(k)'))
            fig.savefig(os.path.join(work_path, 'infer', 'spectrum_case_{}_step_{}.jpg'.format(case_id, time_id)))
            plt.close(fig)


    truth_spectrum = np.stack(truth_spectrum, axis=0)
    truth_spectrum = truth_spectrum.reshape(total_infer_cases, -1, truth_spectrum.shape[-2], truth_spectrum.shape[-1])
    preds_spectrum = np.stack(preds_spectrum, axis=0)
    preds_spectrum = preds_spectrum.reshape(total_infer_cases, -1, preds_spectrum.shape[-2], preds_spectrum.shape[-1])


    for time_id in range(0, total_infer_steps, 50):

        fig, axs = plt.subplots(1, 1, figsize=(10, 5), num=3, constrained_layout=True)
        Visual.plot_value(fig, axs, truth_spectrum[0, 0, :, 0], truth_spectrum[:, time_id//50, :, 1].mean(axis=0), 
                                    std=truth_spectrum[:, time_id//50, :, 1].std(axis=0), label='truth_E(k)',
                                    axis_log=(True, True), xylabels=('k', 'E(k)'))

        Visual.plot_value(fig, axs, preds_spectrum[0, 0, :, 0], preds_spectrum[:, time_id//50, :, 1].mean(axis=0), 
                                    std=preds_spectrum[:, time_id//50, :, 1].std(axis=0), label='preds_E(k)',
                                    axis_log=(True, True), xylabels=('k', 'E(k)'))

        fig.savefig(os.path.join(work_path, 'infer', 'spectrum_mean_step_{}.jpg'.format(time_id)))
        plt.close(fig)