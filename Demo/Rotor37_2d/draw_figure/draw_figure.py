from torch.utils.data import DataLoader
# import torch
# print(torch.__version__)
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
from utilizes_draw import *


if __name__ == "__main__":

    # name = 'FNO_0'
    input_dim = 28
    output_dim = 5
    work_load_path = os.path.join("..", "work_train_MLP")
    workList = os.listdir(work_load_path)
    for name in workList:
        work_path = os.path.join(work_load_path, name)
        work = WorkPrj(work_path)


        nameReal = name.split("_")[0]
        id = None
        if len(name.split("_")) == 2:
            id = int(name.split("_")[1])



        if torch.cuda.is_available():
            Device = torch.device('cuda')
        else:
            Device = torch.device('cpu')

        norm_save_x = work.x_norm
        norm_save_y = work.y_norm

        x_normalizer = DataNormer([1, 1], method="mean-std", axis=0)
        x_normalizer.load(norm_save_x)
        y_normalizer = DataNormer([1, 1], method="mean-std", axis=0)
        y_normalizer.load(norm_save_y)
        if os.path.exists(work.yml):
            Net_model, inference, _, _ = build_model_yml(work.yml, Device, name=nameReal)
            isExist = os.path.exists(work.pth)
            if isExist:
                checkpoint = torch.load(work.pth, map_location=Device)
                Net_model.load_state_dict(checkpoint['net_model'])
        else:
            Net_model, inference = rebuild_model(work_path, Device, name=nameReal)
        train_loader, valid_loader, _, _ = loaddata(nameReal, 2500, 400, shuffled=True)

        for type in ["valid", "train"]:
            if type == "valid":
                true, pred = get_true_pred(valid_loader, Net_model, inference, Device,
                                           name=nameReal, iters=10, alldata=True)
            elif type == "train":
                true, pred = get_true_pred(train_loader, Net_model, inference, Device,
                                           name=nameReal, iters=10, alldata=True)
            true = y_normalizer.back(true)
            pred = y_normalizer.back(pred)

            input_para = {
                "PressureStatic": 0,
                "TemperatureStatic": 1,
                "V2": 2,
                "W2": 3,
                "DensityFlow": 4,
            }

            grid = get_grid(real_path=os.path.join("..", "data"))
            plot_error_box(true, pred, save_path=None, type=type, work_path=work_path)

            post_true = Post_2d(true, grid,
                                inputDict=input_para,
                                )
            post_pred = Post_2d(pred, grid,
                                inputDict=input_para,
                                )

            # parameterList = []
            parameterList = [
                             "Efficiency", "EfficiencyPoly",
                             "PressureRatioV", "TemperatureRatioV",
                             "PressureLossR", "EntropyStatic",
                             "MachIsentropic",
                             "Load",
                             ]
            parameterListN = [
                "PR", "TR",
                "Eff", "EffPoly",
                "PLoss", "Entr",
                "Mach", "Load",
                "MF"]

            dict = plot_error(post_true, post_pred, parameterList + ["MassFlow"],
                       paraNameList=parameterListN,
                       save_path=None, fig_id=0, label=None, work_path=work_path, type=type)

            if type=="valid":
                np.savez(os.path.join(work_path, "FNO_num.npz"), **dict)

            plot_field_2d(post_true, post_pred, parameterList, work_path=work_path, type=type, grid=grid)
            #
            for ii in range(3):
                post_compare = Post_2d(np.concatenate((true[ii:ii + 1, :], pred[ii:ii + 1, :]), axis=0), grid,
                                       inputDict=input_para,
                                       )
                plot_span_curve(post_compare, parameterList,
                                save_path=None, fig_id=ii, label=None, type=type, work_path=work_path)
