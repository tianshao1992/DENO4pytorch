import os

import numpy as np
import torch
from post_process.load_model import loaddata, rebuild_model, build_model_yml
from Utilizes.process_data import DataNormer, MatLoader, SquareMeshGenerator
from Demo.Rotor37_2d.utilizes_rotor37 import get_grid, get_origin
from draw_figure import plot_span_std, plot_span_curve, plot_field_2d
from Utilizes.visual_data import MatplotlibVision
import matplotlib.pyplot as plt
from post_process.post_data import Post_2d
from train_model.model_whole_life import WorkPrj
from run_FNO import feature_transform

if __name__ == "__main__":
    # 数据读入
    work_path = os.path.join("..", "data", "data_collect")
    isCreated = os.path.exists(work_path)
    if not isCreated: os.mkdir(work_path)

    grid = get_grid(real_path=os.path.join("..", "data"))
    design, field = get_origin(realpath=os.path.join("..", "data"), getridbad=False)
                               # quanlityList=["Static Pressure", "Static Temperature",
                               #               'Absolute Total Temperature',  # 'Absolute Total Pressure',
                               #               'Relative Total Temperature',  # 'Relative Total Pressure',
                               #               "DensityFlow",
                               #               # "Vxyz_X", "Vxyz_Y",
                               #               ])
    true = field[:1, :, :, :] #只取一个样本
    pred = true + np.random.rand(*true.shape) * 0.5 - 1
    input_para = {
        "PressureStatic": 0,
        "TemperatureStatic": 1,
        "V2": 2,
        "W2": 3,
        "DensityFlow": 4,
    }
    ii = 0
    post_true = Post_2d(true, grid,
                        inputDict=input_para,
                        )
    post_pred = Post_2d(pred, grid,
                        inputDict=input_para,
                        )
    parameterList = [
                     "PressureRatioV", "TemperatureRatioV",
                     "Efficiency", "EfficiencyPoly",
                     "PressureLossR", "EntropyStatic",
                     "MachIsentropic", "Load"]
    hub_out = 0.1948
    shroud_out = 0.2370
    # print(post_true.span_space_average(post_true.DensityFlow[:, :, -1])*(shroud_out**2-hub_out**2)*np.pi)
    plot_span_curve(post_true, parameterList, work_path=work_path)
    # grid = get_grid(real_path=os.path.join("..", "data"))
    # plot_field_2d(post_true, post_pred, parameterList, grid=grid, work_path=work_path)