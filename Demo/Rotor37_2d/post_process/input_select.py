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

if __name__ == "__main__":
    work_path = os.path.join("..", "data_collect")
    isCreated = os.path.exists(work_path)
    if not isCreated: os.mkdir(work_path)

    grid = get_grid(real_path=os.path.join("..", "data"))
    design, field = get_origin(realpath=os.path.join("..", "data"),
                               quanlityList=["Static Pressure", "Static Temperature", "Density",
                                             'Relative Total Temperature',
                                             "Vxyz_X", "Vxyz_Y",
                                             'Relative Total Pressure',
                                             ])
    true = field[:1, :, :, :-1]
    pred = true + np.random.rand(*true.shape) * 0.5 - 1
    input_para = {
        "PressureStatic": 0,
        "TemperatureStatic": 1,
        "Density": 2,
        # "PressureTotalW": 3,
        "TemperatureTotalW": 3,
        "VelocityX": 4,
        "VelocityY": 5,
    }
    ii = 0
    post_true = Post_2d(true, grid,
                        inputDict=input_para,
                        )
    post_pred = Post_2d(pred, grid,
                        inputDict=input_para,
                        )

    fig_id = 0
    # parameterList = ["Efficiency", "EntropyStatic"]
    cal = post_true.PressureTotalW
    real = field[:1, :, :, -1]

    error = np.abs(cal - real)
    print(np.max(error))

    temp = np.log10(post_true.TemperatureTotalW/post_true.TemperatureStatic) / np.log10(real/post_true.PressureStatic)
    kappa = 1/(1 - temp)
    print(kappa)


