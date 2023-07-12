# 将优化预测结果与CFD验证结果对比
import os
import numpy as np
import yaml
import torch
from tabulate import tabulate
import pandas as pd
from CFD_verify import load_CFD_mat, get_pred_rst


if __name__ == "__main__":
    work_path = os.path.join("..", "work_train_FNO2", "FNO_1") #当前效果最好的模型
    name_opt = "EPM_optimization_tasks"
    name = "FNO"

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')


    mat_path = os.path.join("..","data","sampleRstZip_57.mat")

    #加载CFD结果
    parameterList = [
        "Efficiency",
        "EfficiencyPoly",
        "PressureRatioV",
        "TemperatureRatioV",
        "PressureRatioW",
        "TemperatureRatioW",
        "PressureLossR",
        "EntropyStatic",
        "MachIsentropic",
        "Load",
        "MassFlow"
    ]

    dict_true = load_CFD_mat(mat_path, parameterList)
    #加载预测结果 # 需要加载模型获得全部结果
    ref_data = np.zeros((len(parameterList)))
    for ii, parameter in enumerate(parameterList):
        ref_data[ii] = dict_true[parameter][0]

    save_path = os.path.join("..", "data", "final_fig", 'ref_rst.csv')
    np.savetxt(save_path, ref_data, delimiter=',')



