import numpy as np
import os
import torch
from Utilizes.visual_data import MatplotlibVision
from Utilizes.process_data import DataNormer, MatLoader
import matplotlib.pyplot as plt
from post_process.post_data import Post_2d
from Demo.Rotor37_2d.utilizes_rotor37 import get_grid, get_origin
from post_process.load_model import loaddata, rebuild_model, get_true_pred, build_model_yml
from train_model.model_whole_life import WorkPrj
from Utilizes.loss_metrics import FieldsLpLoss

def load_Npz(npzFile, quanlityList=None):
    loaded_dict = np.load(npzFile)
    if quanlityList is None:
        quanlityList = loaded_dict.files
    fields = np.zeros([64, 64, len(quanlityList)])
    for ii, quanlity in enumerate(quanlityList):
        if quanlity in loaded_dict.files:
            fields[:, :, ii] = loaded_dict[quanlity]

    return fields

def plot_flow_curve(post, parameterList,
                    save_path = None, work_path=None, fig_id = 0,
                    label = None, type='', singlefile=False):
# 绘制单个对象曲线
    if not isinstance(parameterList,list):
        parameterList = [parameterList]

    Visual = MatplotlibVision(work_path, input_name=('Z', 'R'), field_name=('n')) # 不在此处设置名称
    for parameter_Name in parameterList:
        fig, axs = plt.subplots(1, 1, figsize=(6, 3), num=1)
        value_flow = post.field_density_average(parameter_Name, location="whole")

        for ii in range(post.num):
            Visual.plot_value(fig, axs, np.linspace(0,1,post.n_2d), value_flow[ii], label=label,
                              title=parameter_Name, xylabels=(parameter_Name, "z-axis"))

        if save_path is None:
            if singlefile:
                work_path_sub = os.path.join(work_path, parameter_Name)
                isExist = os.path.exists(work_path_sub)
                if not isExist:
                    os.mkdir(work_path_sub)
                jpg_path = os.path.join(work_path_sub, parameter_Name + "_std_" + str(fig_id) + '.jpg')
            else:
                jpg_path = os.path.join(work_path, type + parameter_Name + "_flow_" + str(fig_id) + '.jpg')
        fig.savefig(jpg_path)
        plt.close(fig)

def plot_span_curve(post, parameterList,
                    save_path=None, work_path=None, fig_id=0,
                    label=None, type='', singlefile=False):
    # 绘制单个对象曲线
    if not isinstance(parameterList, list):
        parameterList = [parameterList]

    Visual = MatplotlibVision(work_path, input_name=('Z', 'R'), field_name=('n'))  # 不在此处设置名称
    for parameter_Name in parameterList:
        fig, axs = plt.subplots(1, 1, figsize=(3, 6), num=1)
        value_span = getattr(post, parameter_Name)

        for ii in range(post.num):
            Visual.plot_value(fig, axs, value_span[ii, :, -1], np.linspace(0, 1, post.n_1d), label=label,
                              title=parameter_Name, xylabels=(parameter_Name, "span"))

        # Visual.plot_curve_scatter(fig, axs, value_span[:, :, -1],
        #                   np.tile(np.linspace(0, 1, post.n_1d), (value_span.shape[0], 1)), label=label,
        #                   colorList = ['red', 'b'], markerList=['-', '*'],
        #                   title=parameter_Name, xylabels=(parameter_Name, "span"))

        if save_path is None:
            if singlefile:
                work_path_sub = os.path.join(work_path, parameter_Name)
                isExist = os.path.exists(work_path_sub)
                if not isExist:
                    os.mkdir(work_path_sub)
                jpg_path = os.path.join(work_path_sub, parameter_Name + "_std_" + str(fig_id) + '.jpg')
            else:
                jpg_path = os.path.join(work_path, type + parameter_Name + "_" + str(fig_id) + '.jpg')
        fig.savefig(jpg_path)
        plt.close(fig)


def plot_field_2d(post_true, post_pred, parameterList, save_path=None, work_path=None, fig_id=0, label=None, type='',
                  grid=None):
    # 绘制单个对象曲线
    if not isinstance(parameterList, list):
        parameterList = [parameterList]

    Visual = MatplotlibVision(work_path, input_name=('Z', 'R'), field_name=('n'))  # 不在此处设置名称
    for parameter_Name in parameterList:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), num=1)
        value_field_true = getattr(post_true, parameter_Name)
        value_field_pred = getattr(post_pred, parameter_Name)

        fmin_max = None
        if 'Efficiency' in parameter_Name:
            fmin_max = [[0], [1]]  # 因为函数默认有多个输入，这里是二维数组

        Visual.plot_fields_ms(fig, axs, value_field_true[0, :, :, np.newaxis], value_field_pred[0, :, :, np.newaxis],
                              grid, fmin_max=fmin_max)
        if save_path is None:
            jpg_path = os.path.join(work_path, type + parameter_Name + "_Field_" + str(fig_id) + '.jpg')
        fig.savefig(jpg_path)
        plt.close(fig)


def plot_span_std(post, parameterList,
                  save_path=None, fig_id=0, label=None,
                  work_path=None, rangeIndex=1e2,
                  singlefile=False, singlefigure=False,
                  fig_dict=None, axs_dict=None,
                  ):
    # 绘制一组样本的mean-std分布
    if not isinstance(parameterList, list):
        parameterList = [parameterList]

    Visual = MatplotlibVision(work_path, input_name=('Z', 'R'), field_name=('unset'))  # 不在此处设置名称

    for parameter_Name in parameterList:

        if singlefigure:
            fig = fig_dict[parameter_Name]
            axs = axs_dict[parameter_Name]
        else:
            fig, axs = plt.subplots(1, 1, figsize=(3, 6), num=1)

        value_span = getattr(post, parameter_Name)  # shape = [num, 64, 64]
        value_span = np.mean(value_span[:, :, -10:], axis=2) # shape = [num, 64]
        normalizer = DataNormer(value_span, method='mean-std', axis=0)  # 这里对网格上的具体数据进行平均

        Visual.plot_value_std(fig, axs, normalizer.mean, np.linspace(0, 1, post.n_1d), label=label,
                              std=normalizer.std, rangeIndex=rangeIndex, stdaxis=0,
                              title=parameter_Name, xylabels=(parameter_Name, "span"))
        if not singlefigure:
            if save_path is None:
                if singlefile:
                    work_path_sub = os.path.join(work_path, parameter_Name)
                    isExist = os.path.exists(work_path_sub)
                    if not isExist:
                        os.mkdir(work_path_sub)
                    jpg_path = os.path.join(work_path_sub, parameter_Name + "_std_" + str(fig_id) + '.jpg')
                else:
                    jpg_path = os.path.join(work_path, parameter_Name + "_std_" + str(fig_id) + '.jpg')
            fig.savefig(jpg_path)
            plt.close(fig)

def plot_flow_std(post, parameterList,
                  save_path=None, fig_id=0, label=None,
                  work_path=None, rangeIndex=1e2, singlefile=False):
    # 绘制一组样本的mean-std分布
    if not isinstance(parameterList,list):
        parameterList = [parameterList]

    Visual = MatplotlibVision(work_path, input_name=('Z', 'R'), field_name=('unset')) # 不在此处设置名称
    for parameter_Name in parameterList:
        fig, axs = plt.subplots(1, 1, figsize=(6, 3), num=1)
        # value_span = getattr(post, parameter_Name) # shape = [num, 64, 64]
        value_flow = post.field_density_average(parameter_Name, location="whole")

        normalizer = DataNormer(value_flow, method='mean-std', axis=(0, )) #这里对网格上的具体数据进行平均

        Visual.plot_value_std(fig, axs, np.linspace(0,1,post.n_1d), normalizer.mean, label=label,
                          std=normalizer.std, stdaxis=1, rangeIndex=rangeIndex,
                          title=parameter_Name, xylabels=(parameter_Name, "span"))

        if save_path is None:
            if singlefile:
                work_path_sub = os.path.join(work_path, parameter_Name)
                isExist = os.path.exists(work_path_sub)
                if not isExist:
                    os.mkdir(work_path_sub)
                jpg_path = os.path.join(work_path_sub, parameter_Name + "_std_" + str(fig_id) + '.jpg')
            else:
                jpg_path = os.path.join(work_path, parameter_Name + "_flow_std_" + str(fig_id) + '.jpg')
        fig.savefig(jpg_path)
        plt.close(fig)



def plot_error(post_true, post_pred, parameterList,
               save_path = None, fig_id = 0, label=None, work_path=None, type=None, paraNameList = None):
    """
    针对某个对象0维性能参数，绘制预测误差表示图
    """
    dict_save = {}
    if not isinstance(parameterList, list):
        parameterList = [parameterList]
    Err_all = []
    if paraNameList is None:
        paraNameList = ('n')
    Visual = MatplotlibVision(work_path, input_name=('Z', 'R'), field_name=paraNameList) # 不在此处设置名称
    for parameter_Name in parameterList:
        fig, axs = plt.subplots(1, 1, figsize=(10, 10), num=1)

        if parameter_Name in ("MassFlow"):
            value_span_true = post_true.get_MassFlow()
            value_span_pred = post_pred.get_MassFlow()


        else:
            value_span_true = getattr(post_true, parameter_Name) #shape[num, 64, 64]
            value_span_true = post_true.span_density_average(value_span_true[:, :, -1]) # shape[num, 1]
            value_span_pred = getattr(post_pred, parameter_Name)
            value_span_pred = post_true.span_density_average(value_span_pred[:, :, -1])

        Visual.plot_regression(fig, axs, value_span_true.squeeze(), value_span_pred.squeeze(),
                               title=parameter_Name)

        if save_path is None:
            jpg_path = os.path.join(work_path, type + parameter_Name + "_error_" + str(fig_id) + '.jpg')
        fig.savefig(jpg_path)
        plt.close(fig)

        dict_save.update({parameter_Name : value_span_pred})
        Err = np.abs((value_span_pred - value_span_true) / value_span_true)
        Err_all.append(Err)

    Err_all = np.concatenate(Err_all, axis=1)
    fig, axs = plt.subplots(1, 1, figsize=(10, 10), num=1)
    Visual.plot_box(fig, axs, Err_all, xticks=Visual.field_name)

    jpg_path = os.path.join(work_path, type + "_error-box.jpg")
    fig.savefig(jpg_path)
    plt.close(fig)

    return dict_save

def plot_error_box(true, pred, save_path=None, type=None, work_path = None):
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('Ps', 'Ts', 'rhoV', 'Pt', 'Tt'))
    Error_func = FieldsLpLoss(size_average=False)

    # Error_func.p = 1
    # ErrL1a = Error_func.abs(valid_pred, valid_true)
    # ErrL1r = Error_func.rel(valid_pred, valid_true)
    Error_func.p = 2
    ErrL2a = Error_func.abs(pred, true)
    ErrL2r = Error_func.rel(pred, true)

    fig, axs = plt.subplots(1, 1, figsize=(10, 10), num=1)
    Visual.plot_box(fig, axs, ErrL2r, xticks=Visual.field_name)

    if save_path is None:
        jpg_path = os.path.join(work_path, type + "box.jpg")
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
    parameterList = ["Efficiency", "EntropyStatic", ""]
    plot_error(post_true, post_pred, parameterList, save_path=None, fig_id=0, label=None)