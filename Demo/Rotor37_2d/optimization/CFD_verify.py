# 将优化预测结果与CFD验证结果对比
import os
import numpy as np
import yaml
import torch
from utilizes_rotor37 import get_quanlity_from_mat, get_grid
from post_process.post_data import Post_2d
from train_model.model_whole_life import WorkPrj
from Utilizes.process_data import DataNormer
from post_process.load_model import build_model_yml
from post_process.model_predict import DLModelPost
from tabulate import tabulate
import pandas as pd

def load_CFD_mat(sample_file, parameterList):

    quanlityList = ["Static Pressure", "Static Temperature",
                    'V2', 'W2', "DensityFlow"]

    grid = get_grid(real_path=os.path.join("..", "data"))
    design, fields = get_quanlity_from_mat(sample_file, quanlityList)
    post_true = Post_2d(fields, grid,
                        inputDict=None,
                        )
    all_dict = {}
    for parameter in parameterList:
        if parameter=="MassFlow":
            scalar = post_true.get_MassFlow()
        else:
            value_span = getattr(post_true, parameter)
            scalar = post_true.span_density_average(value_span[:, :, -1])
        all_dict.update({parameter: scalar})

    all_dict.update({"Design": design})

    return all_dict

def get_pred_rst(work_path, name, x, Device):
    work = WorkPrj(work_path)

    norm_save_x = work.x_norm
    norm_save_y = work.y_norm

    x_normlizer = DataNormer([1, 1], method="mean-std", axis=0)
    x_normlizer.load(norm_save_x)
    y_normlizer = DataNormer([1, 1], method="mean-std", axis=0)
    y_normlizer.load(norm_save_y)

    Net_model, inference, _, _ = build_model_yml(work.yml, Device, name=name)
    isExist = os.path.exists(work.pth)
    if isExist:
        checkpoint = torch.load(work.pth, map_location=Device)
        Net_model.load_state_dict(checkpoint['net_model'])

    Net_model.eval()

    model_all = DLModelPost(Net_model, Device,
                            name=name,
                            in_norm=x_normlizer,
                            out_norm=y_normlizer,
                            )

    pred = model_all.predictor_value(x, parameterList=parameterList, setOpt=False)

    return pred


if __name__ == "__main__":
    work_path = os.path.join("..", "work_train_FNO2", "FNO_1") #当前效果最好的模型
    name_opt = "sin_obj_minimize"
    name = "FNO"

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
    #加载CFD结果
    parameterList = [
        "PressureRatioV", "TemperatureRatioV",
        "PressureRatioW", "TemperatureRatioW",
        "Efficiency", "EfficiencyPoly",
        "PressureLossR", "EntropyStatic",
        "MachIsentropic", "Load", "MassFlow"]

    dict_true = load_CFD_mat(mat_path, parameterList)
    #加载预测结果 # 需要加载模型获得全部结果
    pred = get_pred_rst(work_path, name, dict_true["Design"], Device)
    #对比
    # dict_compare_list = []
    dict_compare = {}
    for ii, parameter in enumerate(parameterList):
        compareList = np.concatenate((dict_true[parameter], pred[:, ii:ii+1],
                                      np.abs((dict_true[parameter]-pred[:, ii:ii+1])/dict_true[parameter])),axis=1)
        compareList = np.round(compareList, decimals=5)
        dict_compare.update({parameter : compareList.tolist()})

    table = tabulate(dict_compare, headers='keys', tablefmt='psql')
    print(table)
    df = pd.DataFrame(dict_compare)
    df.to_excel(xls_path, index=range(len(dict_compare['Load'])))



