# 将优化预测结果与CFD验证结果对比
import os
import numpy as np
import yaml
import torch
from tabulate import tabulate
import pandas as pd
from CFD_verify import load_CFD_mat, load_CFD_mat_train
from draw_figure.draw_sensitive import predicter_loader
# from draw_figure.utilizes_draw import plot_span_curve_marker
# from train_model.model_whole_life import WorkPrj
# from Utilizes.process_data import DataNormer
# from post_process.load_model import build_model_yml
# from Demo.Rotor37_2d.utilizes_rotor37 import get_grid, get_origin
# from post_process.post_data import Post_2d
from draw_figure.utilizes_draw import *

def get_predict(work_path, input, name, Device, parameterList):
    parameter_dict = {}
    work = WorkPrj(work_path)

    norm_save_x = work.x_norm
    norm_save_y = work.y_norm

    x_normlizer = DataNormer(np.ndarray([1, 1]), method="mean-std", axis=0)
    x_normlizer.load(norm_save_x)
    y_normlizer = DataNormer(np.ndarray([1, 1]), method="mean-std", axis=0)
    y_normlizer.load(norm_save_y)

    if os.path.exists(work.yml):
        Net_model, inference, _, _ = build_model_yml(work.yml, Device, name=name)
        isExist = os.path.exists(work.pth)
        if isExist:
            checkpoint = torch.load(work.pth, map_location=Device)
            Net_model.load_state_dict(checkpoint['net_model'])
    else:
        Net_model, inference = rebuild_model(work_path, Device, name=name)

    Net_model.eval()

    input = x_normlizer.norm(input)
    pred = predicter_loader(Net_model, input, Device, name=name)

    pred = pred.reshape([pred.shape[0], 64, 64, 5])
    pred = y_normlizer.back(pred)

    input_para = {
        "PressureStatic": 0,
        "TemperatureStatic": 1,
        "V2": 2,
        "W2": 3,
        "DensityFlow": 4,
    }
    grid = get_grid(real_path=os.path.join("..", "data"))
    post_pred = Post_2d(pred.detach().cpu().numpy(), grid,
                        inputDict=input_para,
                        )

    for parameter in parameterList:
        if parameter=="MassFlow":
            scalar = post_pred.get_MassFlow()
        else:
            value_span = getattr(post_pred, parameter)
            scalar = post_pred.span_density_average(value_span[:, :, -1])
        parameter_dict.update({parameter: scalar})

    return parameter_dict



def find_closest_row(A, B):
    min_distance = np.inf
    closest_row_idx = None

    for i, row in enumerate(A):
        distance = np.linalg.norm(row - B)
        if distance < min_distance:
            min_distance = distance
            closest_row_idx = i

    return closest_row_idx, min_distance


if __name__ == "__main__":
    work_path = os.path.join("..", "work", "Trunk_TRA") #当前效果最好的模型
    name_opt = "EPM_optimization_tasks_20230702"
    name = "Transformer"

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    data_path = os.path.join("..", "data", "opt_data")
    npz_path = os.path.join(data_path, name_opt + ".npz")
    yml_path = os.path.join(data_path, name_opt + ".yml")
    txt_path = os.path.join(data_path, name_opt + ".txt")
    mat_path = os.path.join(data_path, name_opt + ".mat")
    xls_path = os.path.join(data_path, name_opt + ".xlsx")

    #提取要分析的任务
    loaded_dict = np.load(npz_path)
    for task_num in [0]:#range(1):
        name_task = "task_" + str(task_num) + "_sample"
        sample = loaded_dict[name_task]

        #加载CFD结果
        parameterList = [
            "PressureRatioV", "TemperatureRatioV",
            "PressureRatioW", "TemperatureRatioW",
            "Efficiency", "EfficiencyPoly",
            "PressureLossR", "EntropyStatic",
            # "MachIsentropic", "Load",
            "MassFlow",
        ]

        dict_true = load_CFD_mat(mat_path, parameterList)
        dict_base = load_CFD_mat(os.path.join("..", "data", "sampleRstZip_57"), parameterList)

        dict_true_train = load_CFD_mat_train(parameterList)


        #匹配任务中样本
        idx_list = []
        for ii in range(sample.shape[0]):
            closest_row_idx, min_distance = find_closest_row(dict_true["Design"], sample[ii])
            # if dict_true["Efficiency"][closest_row_idx] <0.871:
            idx_list.append(closest_row_idx)
            if min_distance>1e-4:
                print("sample is not found!")

        pred = get_predict(work_path, dict_true["Design"][idx_list], name, Device, parameterList)

        idx_new_list = []
        for ii in range(len(idx_list)):
            if pred["Efficiency"][ii] <0.871:
                if pred["PressureRatioV"][ii] >2.01:
                    idx_new_list.append(idx_list[ii])
        idx_list = idx_new_list
        pred = get_predict(work_path, dict_true["Design"][idx_list], name, Device, parameterList)

        # pred_train = get_predict(work_path, dict_true_train["Design"], name, Device, parameterList)

        # idx_list = idx_list[:5]
        #加载预测结果 # 需要加载模型获得全部结果
        true = {}
        true_train = {}
        true_base = {}
        for parameter in parameterList:
            true.update({parameter: dict_true[parameter][idx_list]})
            true_train.update({parameter: dict_true_train[parameter]})
            true_base.update({parameter: dict_base[parameter][0]})



        #绘制前锋图
        Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('p', 't', 'V', 'W', 'mass'))
        fig, axs = plt.subplots(1, 1, figsize=(10, 10), num=1)


        label_1 = "Efficiency"
        # label_2 = "PressureRatioV"
        label_2 = "MassFlow"
        # Visual.plot_curve_scatter(fig, axs, pred_train[label_1].T, pred_train[label_2].T, label=None,
        #                           colorList=['g'], markerList=['.'],
        #                           msList=[70], mfcList=['g'], mecList=['k'],
        #                           title="pareto", xylabels=("span"))

        Visual.plot_curve_scatter(fig, axs, true_train[label_1].T, true_train[label_2].T, labelList=['train-cfd'],
                                  colorList=['steelblue'], markerList=['.'],
                                  msList=[140], mfcList=['steelblue'], mecList=['k'],
                                  title=None, xylabels=("span"))

        Visual.plot_curve_scatter(fig, axs, true_base[label_1], true_base[label_2], labelList=['baseline-cfd'],
                                  colorList=[None], markerList=['^'],
                                  msList=[360], mfcList=['gold'], mecList=['k'],
                                  title=None, xylabels=("span"))

        Visual.plot_curve_scatter(fig, axs, pred[label_1].T, pred[label_2].T, labelList=['optimal-pred'],
                                  colorList=['m'], markerList=['*'],
                                  msList=[360], mfcList=['m'], mecList=['k'],
                                  title=None, xylabels=("span"))

        Visual.plot_curve_scatter(fig, axs, true[label_1].T, true[label_2].T, labelList=['optimal-cfd'],
                                  colorList=['crimson'], markerList=['*'],
                                  msList=[360], mfcList=['crimson'], mecList=['k'],
                                  title=None, xylabels=(["efficiency","mass flow"]))
        axs.legend(loc="lower right", framealpha=0)
        # axs.legend('')
        # axs.set_xlabel('')
        # axs.set_ylabel('')
        # axs.axis('off')
        # plt.show()

        save_path = os.path.join("..", "data", "final_fig")
        jpg_path = os.path.join(save_path, "multi_opt_task_mass_" + str(task_num) + '.jpg')

        fig.savefig(jpg_path)
        plt.close(fig)




