import os

import numpy as np
import torch
from post_process.load_model import loaddata, rebuild_model, build_model_yml
from Utilizes.process_data import DataNormer, MatLoader, SquareMeshGenerator
from Demo.Rotor37_2d.utilizes_rotor37 import get_grid, get_origin
from draw_figure import plot_span_std
from Utilizes.visual_data import MatplotlibVision
import matplotlib.pyplot as plt
from post_process.post_data import Post_2d
from train_model.model_whole_life import WorkPrj
from run_FNO import feature_transform

def predicter(netmodel, input, Device, name=None):
    """
    加载完整的模型预测输入的坐标
    Net_model 训练完成的模型
    input 模型的输入 shape:[num, input_dim]
    """
    if name in ("FNO", "UNet", "Transformer"):
        input = torch.tensor(np.tile(input[:, None, None, :], (1, 64, 64, 1)))
        input = input.to(Device)
        grid = feature_transform(input)
        pred = netmodel(input, grid)
    else:
        input = input.to(Device)
        pred = netmodel(input)

    return pred

def mesh_sliced(input_dim, slice_index, space=None, mesh_size=None, elite=None):
    slice_dim = len(slice_index)
    if space is None:
        space = np.tile(np.array([0, 1]), [slice_dim, 1])
    if mesh_size is None:
        mesh_size = np.ones([slice_dim]) * 101
    if elite is None:
        elite = np.ones([input_dim]) * 0.5
    meshgenerator = SquareMeshGenerator(space, mesh_size)
    slice_grid = meshgenerator.get_grid()
    sample_grid = np.tile(elite, [slice_grid.shape[0],1])

    sample_grid[:, slice_index] = slice_grid

    return torch.tensor(sample_grid, dtype=torch.float)

if __name__ == "__main__":
    name = 'FNO_0'
    input_dim = 28
    output_dim = 5
    work_load_path = os.path.join("..", "work_train_1")
    work_path = os.path.join(work_load_path, name)
    work = WorkPrj(work_path)

    for name in ['FNO_0']:  # workList:#['MLP','deepONet','FNO','UNet','Transformer']:
        nameReal = name.split("_")[0]
        id = None
        if len(name.split("_")) == 2:
            id = int(name.split("_")[1])

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    for ii in range(28):
        sample_grid = mesh_sliced(input_dim, [ii])

        if os.path.exists(work.x_norm):
            norm_save_x = work.x_norm
            norm_save_y = work.y_norm
        else:
            norm_save_x = os.path.join("..", "data", "x_norm_1250.pkl")
            norm_save_y = os.path.join("..", "data", "y_norm_1250.pkl")

        x_normlizer = DataNormer([1, 1], method="mean-std", axis=0)
        x_normlizer.load(norm_save_x)
        sample_grid = x_normlizer.norm(sample_grid)

        if os.path.exists(work.yml):
            Net_model, inference, _, _ = build_model_yml(work.yml, Device, name=nameReal)
            isExist = os.path.exists(work.pth)
            if isExist:
                checkpoint = torch.load(work.pth, map_location=Device)
                Net_model.load_state_dict(checkpoint['net_model'])
        else:
                Net_model, inference = rebuild_model(work_path, Device, name=nameReal)

        Net_model.eval()

        pred = predicter(Net_model, sample_grid, Device, name=nameReal) # 获得预测值


        y_normlizer = DataNormer([1, 1], method="mean-std", axis=0)

        y_normlizer.load(norm_save_y)
        pred = pred.reshape([pred.shape[0], 64, 64, output_dim])
        pred = y_normlizer.back(pred)

        input_para = {
            "PressureStatic": 0,
            "TemperatureStatic": 1,
            "DensityFlow": 2,
            "PressureTotalW": 3,
            "TemperatureTotalW": 4,
        }
        grid = get_grid(real_path=os.path.join("..", "data"))
        post_pred = Post_2d(pred.detach().cpu().numpy(), grid,
                            inputDict=input_para,
                            )

        fig_id = 0

        parameterList = ["PressureLossR"]
        plot_span_std(post_pred, parameterList, work_path=work_path, fig_id=ii)

