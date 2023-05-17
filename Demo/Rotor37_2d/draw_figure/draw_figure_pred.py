import os
import numpy as np
import torch
from post_process.load_model import loaddata, rebuild_model, build_model_yml
from Utilizes.process_data import DataNormer, MatLoader, SquareMeshGenerator
from Demo.Rotor37_2d.utilizes_rotor37 import get_grid, get_origin
from draw_figure import plot_span_std, plot_span_curve, plot_flow_curve, plot_flow_std
from Utilizes.visual_data import MatplotlibVision
import matplotlib.pyplot as plt
from post_process.post_data import Post_2d
from train_model.model_whole_life import WorkPrj
from run_FNO import feature_transform
from SALib.sample import latin



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

def mesh_sliced(input_dim, slice_index, elite=None, type='lhsdesign', sample_num=None):
    slice_dim = len(slice_index)
    if elite is None:
        elite = np.ones([input_dim]) * 0.5
    if sample_num is None:
        sample_num = slice_dim * 101
    if type == "lhsdesign":
        slice_grid = LHSdesign(sample_num, slice_dim)
    elif type == "meshdesign":
        slice_grid = SquareMeshdesign(slice_dim)

    sample_grid = np.tile(elite, [slice_grid.shape[0],1])

    sample_grid[:, slice_index] = slice_grid

    return torch.tensor(sample_grid, dtype=torch.float)

def LHSdesign(sam_num, sam_dim):
    problem = {
        'num_vars': sam_dim,  # 参数数量
        'names': [f'x{i}' for i in range(1, sam_dim + 1)],  # 参数名称
        'bounds': [[0, 1]] * sam_dim,  # 参数范围
    }
    # 生成 LHS 设计样本
    samples = latin.sample(problem, sam_num)

    return samples

def SquareMeshdesign(slice_dim, space=None, mesh_size=None):
    if space is None:
        space = np.tile(np.array([0, 1]), [slice_dim, 1])
    if mesh_size is None:
        mesh_size = np.ones([slice_dim]) * 21

    meshgenerator = SquareMeshGenerator(space, mesh_size)
    slice_grid = meshgenerator.get_grid()

    return  slice_grid


if __name__ == "__main__":
    name = 'FNO_1'
    input_dim = 28
    output_dim = 5
    work_load_path = os.path.join("..", "work_train_FNO2")
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

    if os.path.exists(work.x_norm):
        norm_save_x = work.x_norm
        norm_save_y = work.y_norm
    else:
        norm_save_x = os.path.join("..", "data", "x_norm_1250.pkl")
        norm_save_y = os.path.join("..", "data", "y_norm_1250.pkl")

    x_normlizer = DataNormer([1, 1], method="mean-std", axis=0)
    x_normlizer.load(norm_save_x)

    if os.path.exists(work.yml):
        Net_model, inference, _, _ = build_model_yml(work.yml, Device, name=nameReal)
        isExist = os.path.exists(work.pth)
        if isExist:
            checkpoint = torch.load(work.pth, map_location=Device)
            Net_model.load_state_dict(checkpoint['net_model'])
    else:
        Net_model, inference = rebuild_model(work_path, Device, name=nameReal)

    Net_model.eval()

    # var_group = list(range(28))
    # var_group = [[x] for x in var_group]

    #按叶高分组
    # var_group = [
    #             [0, 1, 2],
    #             [3, 4, 5, 6, 7],
    #             [8, 9, 10, 11, 12],
    #             [13, 14, 15, 16, 17],
    #             [18, 19, 20, 21, 22],
    #             [23, 24, 25, 26, 27]
    #             ]

    #按流向位置分组
    # var_group = [
    #     [0, 1, 2],
    #     [3, 8, 13, 18, 23],
    #     [4, 9, 14, 19, 24],
    #     [5, 10, 15, 20, 25],
    #     [6, 11, 16, 21, 26],
    #     [7, 12, 17, 22, 27],
    # ]

    # 按叶高分组-只有前缘
    # var_group = [
    #             [0, 1, 2],
    #             [3,4],
    #             [8,9],
    #             [13,14],
    #             [18,19],
    #             [23,24]
    #             ]

    # 按叶高分组-只有中间
    # var_group = [
    #             [0, 1, 2],
    #             [5],
    #             [10],
    #             [15],
    #             [20],
    #             [25]
    #             ]

    # 按叶高分组-只有尾缘
    var_group = [
                [0, 1, 2],
                [6, 7],
                [11, 12],
                [16, 17],
                [21, 22],
                [26, 27]
                ]

    for idx, var_list in enumerate(var_group):
        sample_grid = mesh_sliced(input_dim, var_list, sample_num=len(var_list)*31)
        sample_grid = x_normlizer.norm(sample_grid)

        pred = predicter(Net_model, sample_grid, Device, name=nameReal) # 获得预测值
        y_normlizer = DataNormer([1, 1], method="mean-std", axis=0)
        y_normlizer.load(norm_save_y)
        pred = pred.reshape([pred.shape[0], 64, 64, output_dim])
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

        fig_id = 0

        parameterList = [
                         # "Efficiency", "EfficiencyPoly",
                         "PressureRatioV", "TemperatureRatioV",
                         "PressureLossR", "EntropyStatic",
                         "MachIsentropic", "Load",
                         ]

        save_path = os.path.join(work_path,"sensitive")
        isExist = os.path.exists(save_path)
        if not isExist:
            os.mkdir(save_path)
        plot_span_std(post_pred, parameterList, work_path=save_path, fig_id=idx, rangeIndex=10)
        # plot_span_curve(post_pred, parameterList, work_path=save_path, fig_id=idx)
        plot_flow_std(post_pred, parameterList, work_path=save_path, fig_id=idx, rangeIndex=10)
        # plot_flow_curve(post_pred, parameterList, work_path=save_path, fig_id=idx)
        pred = None

