from torch.utils.data import DataLoader
# import torch
# print(torch.__version__)
import numpy as np
import os
import torch
from Utilizes.visual_data import MatplotlibVision
from Utilizes.process_data import DataNormer, MatLoader
import matplotlib.pyplot as plt
from post_data import Post_2d
from Demo.Rotor37_2d.utilizes_rotor37 import get_grid, get_origin
from post_process.load_model import loaddata, rebuild_model, get_true_pred


def load_Npz(npzFile, quanlityList=None):
    loaded_dict = np.load(npzFile)
    if quanlityList is None:
        quanlityList = loaded_dict.files
    fields = np.zeros([64, 64, len(quanlityList)])
    for ii, quanlity in enumerate(quanlityList):
        if quanlity in loaded_dict.files:
            fields[:,:,ii] = loaded_dict[quanlity]

    return fields

def plot_span_curve(post, parameterList, save_path = None, fig_id = 0, label = None):
# 绘制单个对象曲线
    if not isinstance(parameterList,list):
        parameterList = [parameterList]

    Visual = MatplotlibVision(work_path, input_name=('Z', 'R'), field_name=('unset')) # 不在此处设置名称
    for parameter_Name in parameterList:
        fig, axs = plt.subplots(1, 1, figsize=(3, 6), num=1)
        value_span = getattr(post, parameter_Name)

        for ii in range(post.num):
            Visual.plot_value(fig, axs, value_span[ii, :, -1], np.linspace(0,1,post.n_1d), label=label,
                              title=parameter_Name, xylabels=(parameter_Name, "span"))

        if save_path is None:
            jpg_path = os.path.join(work_path, parameter_Name + "_" + str(fig_id) + '.jpg')
        fig.savefig(jpg_path)
        plt.close(fig)


def plot_span_std(post, parameterList, save_path=None, fig_id=0, label=None, work_path=None):
    # 绘制一组样本的mean-std分布
    if not isinstance(parameterList,list):
        parameterList = [parameterList]

    Visual = MatplotlibVision(work_path, input_name=('Z', 'R'), field_name=('unset')) # 不在此处设置名称
    for parameter_Name in parameterList:
        fig, axs = plt.subplots(1, 1, figsize=(3, 6), num=1)
        value_span = getattr(post, parameter_Name) # shape = [num, 64, 64]

        normalizer = DataNormer(value_span, method='mean-std', axis=(0, )) #这里对网格上的具体数据进行平均

        Visual.plot_value_std(fig, axs, normalizer.mean[:, -1], np.linspace(0,1,post.n_1d), label=label,
                          std=normalizer.std[:,-1], rangeIndex=1e2, stdaxis=0,
                          title=parameter_Name, xylabels=(parameter_Name, "span"))

        if save_path is None:
            jpg_path = os.path.join(work_path, parameter_Name + "_std_" + str(fig_id) + '.jpg')
        fig.savefig(jpg_path)
        plt.close(fig)


def plot_error(post_true, post_pred, parameterList,
               save_path = None, fig_id = 0, label=None, work_path=None):
    """
    针对某个对象0维性能参数，绘制预测误差表示图
    """
    if not isinstance(parameterList,list):
        parameterList = [parameterList]

    Visual = MatplotlibVision(work_path, input_name=('Z', 'R'), field_name=('unset')) # 不在此处设置名称
    for parameter_Name in parameterList:
        fig, axs = plt.subplots(1, 1, figsize=(10, 10), num=1)
        value_span_true = getattr(post_true, parameter_Name) #shape[num, 64, 64]
        value_span_true = post_true.span_density_average(value_span_true[:, :, -1]) # shape[num, 1]
        value_span_pred = getattr(post_pred, parameter_Name)
        value_span_pred = post_true.span_density_average(value_span_pred[:, :, -1])

        Visual.plot_regression(fig, axs, value_span_true.squeeze(), value_span_pred.squeeze(),
                          title=parameter_Name)

        if save_path is None:
            jpg_path = os.path.join(work_path, parameter_Name + "_error_" + str(fig_id) + '.jpg')
        fig.savefig(jpg_path)
        plt.close(fig)


def plot_saliency_map():
    """
    对于28个变量的贡献度进行分析
    """

def a_case():
    work_path = os.path.join("..", "data_collect")
    isCreated = os.path.exists(work_path)
    if not isCreated: os.mkdir(work_path)

    grid = get_grid(real_path=os.path.join("..", "data"))
    design, field = get_origin(realpath=os.path.join("..", "data"),
                               quanlityList=["Static Pressure", "Static Temperature", "Density",
                                             'Relative Total Pressure', 'Relative Total Temperature',
                                             "Vxyz_X", "Vxyz_Y",
                                             ])
    true = field
    pred = true + np.random.rand(*true.shape) * 0.5 - 1
    input_para = {
        "PressureStatic": 0,
        "TemperatureStatic": 1,
        "Density": 2,
        "PressureTotalW": 3,
        "TemperatureTotalW": 4,
        "VelocityX": 5,
        "VelocityY": 6,
    }
    ii = 0
    post_true = Post_2d(true, grid,
                        inputDict=input_para,
                        )
    post_pred = Post_2d(pred, grid,
                        inputDict=input_para,
                        )

    fig_id = 0
    parameterList = ["Efficiency", "EntropyStatic"]
    plot_error(post_true, post_pred, parameterList, save_path=None, fig_id=0, label=None)



if __name__ == "__main__":
    work_path = os.path.join("..", "data_collect")
    isCreated = os.path.exists(work_path)
    if not isCreated: os.mkdir(work_path)

    name = 'MLP'
    input_dim = 28
    output_dim = 5
    work_load_path = os.path.join("..", "work")
    work_path = os.path.join(work_load_path, name)

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    x_normlizer = DataNormer([1, 1], method="mean-std", axis=0)
    norm_save_x = os.path.join("work_path", "x_norm.pkl")
    x_normlizer.load(norm_save_x)
    y_normlizer = DataNormer([1, 1], method="mean-std", axis=0)
    norm_save_y = os.path.join("work_path", "y_norm.pkl")
    y_normlizer.load(norm_save_y)

    Net_model, inference = rebuild_model(work_path, Device, name=name)
    train_loader, valid_loader, x_normalizer, y_normalizer = loaddata(name, 2700, 250)

    # train_true, train_pred = get_true_pred(train_loader, Net_model, inference, Device)
    # valid_true, valid_pred = get_true_pred(valid_loader, Net_model, inference, Device, name=name)
    # valid_true = y_normalizer.back(valid_true)
    # valid_pred = y_normalizer.back(valid_pred)

    train_true, train_pred = get_true_pred(train_loader, Net_model, inference, Device, name=name)
    train_true = y_normalizer.back(train_true)
    train_pred = y_normalizer.back(train_pred)


    true = train_true
    pred = train_pred
    input_para = {
        "PressureStatic": 0,
        "TemperatureStatic": 1,
        "DensityFlow": 2,
        "PressureTotalW": 3,
        "TemperatureTotalW": 4,
    }
    ii = 0
    grid = get_grid(real_path=os.path.join("..", "data"))
    post_true = Post_2d(true, grid,
                        inputDict=input_para,
                        )
    post_pred = Post_2d(pred, grid,
                        inputDict=input_para,
                        )

    fig_id = 0
    parameterList = ["PressureLossR", "EntropyStatic"]
    plot_error(post_true, post_pred, parameterList, save_path=None, fig_id=0, label=None, work_path=work_path)





