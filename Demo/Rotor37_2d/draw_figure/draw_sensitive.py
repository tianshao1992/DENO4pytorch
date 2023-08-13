import os
import numpy as np
import torch
from post_process.load_model import loaddata, rebuild_model, build_model_yml
from Utilizes.process_data import DataNormer, MatLoader, SquareMeshGenerator
from Demo.Rotor37_2d.utilizes_rotor37 import get_grid, get_origin
from draw_figure.utilizes_draw import plot_flow_std, plot_span_std
# from utilizes_draw import plot_span_std, plot_span_curve, plot_flow_curve, plot_flow_std
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

def predicter_loader(netmodel, input_data, Device, name=None):
    """
    加载完整的模型预测输入的坐标
    Net_model 训练完成的模型
    input 模型的输入 shape:[num, input_dim]
    先转换数据，分批计算
    """
    # torch.utils.data.TensorDataset(input_data)
    loader = torch.utils.data.DataLoader(input_data,
                                         batch_size=32,
                                         shuffle=False,
                                         drop_last=False)
    pred = []
    for input in loader:
        if name in ("FNO", "UNet"):
            with torch.no_grad():
                input = torch.tensor(np.tile(input[:, None, None, :], (1, 64, 64, 1)))
                input = input.to(Device)
                grid = feature_transform(input)
                temp = netmodel(input, grid)
                pred.append(temp.clone())
                temp = None

        elif name in ("Transformer"):
            from Utilizes.geometrics import gen_uniform_grid
            grid = gen_uniform_grid(torch.tensor(np.zeros([1, 64, 64, 5]))).to(Device)
            xx = input.to(Device)
            coords = grid.tile([xx.shape[0], 1, 1, 1])

            temp = netmodel(xx, coords)
            pred.append(temp.clone())
            temp = None

        else:
            with torch.no_grad():
                input = input.to(Device)
                temp = netmodel(input)
                pred.append(temp.clone())
                temp = None

    pred = torch.cat(pred, dim=0)
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

def MkdirCheck(file_path):
    isExist = os.path.exists(file_path)
    if not isExist:
        os.mkdir(file_path)


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


    norm_save_x = work.x_norm
    norm_save_y = work.y_norm

    x_normlizer = DataNormer([1, 1], method="mean-std", axis=0)
    x_normlizer.load(norm_save_x)
    y_normlizer = DataNormer([1, 1], method="mean-std", axis=0)
    y_normlizer.load(norm_save_y)

    if os.path.exists(work.yml):
        Net_model, inference, _, _ = build_model_yml(work.yml, Device, name=nameReal)
        isExist = os.path.exists(work.pth)
        if isExist:
            checkpoint = torch.load(work.pth, map_location=Device)
            Net_model.load_state_dict(checkpoint['net_model'])
    else:
        Net_model, inference = rebuild_model(work_path, Device, name=nameReal)

    Net_model.eval()

    parameterList = [
        # "Efficiency",
        # "EfficiencyPoly",
        # "PressureRatioV",
        # "TemperatureRatioV",
        "PressureLossR",
        # "EntropyStatic",
        # "MachIsentropic",
        # "Load",
    ]

    var_group = list(range(25))
    var_group = [[x] for x in var_group]


    dict_fig = {}
    dict_axs = {}
    for ii, parameter in enumerate(parameterList):
        fig, axs = plt.subplots(5, 5, figsize=(15, 15), num=ii)
        dict_fig.update({parameter: fig})
        dict_axs.update({parameter: axs})


    for idx, var_list in enumerate(var_group):
        sample_grid = mesh_sliced(input_dim, var_list, sample_num=1001)
        sample_grid = x_normlizer.norm(sample_grid)

        pred = predicter_loader(Net_model, sample_grid, Device, name=nameReal) # 获得预测值

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



        # save_path = os.path.join(work_path, "sensitive_test")
        save_path = os.path.join("..", "data", "final_fig")

        # MkdirCheck(save_path)
        # MkdirCheck(os.path.join(save_path, "span_std"))
        dict_axs_sub = {}

        #old order way
        # x1 = int((24-idx) / 5)
        # x2 = (24-idx) % 5

        # new order way
        x1 = int(idx / 5)
        x2 = (24-idx) % 5

        for parameter in parameterList:
            dict_axs_sub.update({parameter : dict_axs[parameter][x1][x2]})

        xlimList = [
            [0.5, 1.0],
            [-0.04,0.16],
            [1.75, 2.2],
            [0, 115],
        ]
        tt = 1
        # plot_span_std(post_pred, parameterList,
        #               work_path=os.path.join(save_path, "span_std"),
        #               fig_id=idx, rangeIndex=50, singlefile=True, xlim=xlimList[tt],
        #               singlefigure=True, fig_dict=dict_fig, axs_dict=dict_axs_sub)

        # plot_span_std(post_pred, parameterList,
        #               work_path=os.path.join(save_path, "span_std"),
        #               fig_id=idx, rangeIndex=50, singlefile=True,
        #               singlefigure=True, fig_dict=dict_fig, axs_dict=dict_axs_sub)

        plot_flow_std(post_pred, parameterList,
                      work_path=os.path.join(save_path, "flow_std"),
                      fig_id=idx, rangeIndex=40, singlefile=True, xlim=xlimList[tt],
                      singlefigure=True, fig_dict=dict_fig, axs_dict=dict_axs_sub)

        # for ii, parameter in enumerate(parameterList):
        #     fig = dict_fig[parameter]
        #     plt.figure(fig.number)
        #     plt.show()

        # MkdirCheck(os.path.join(save_path, "span_curve"))
        # plot_span_curve(post_pred, parameterList,
        #                 work_path=os.path.join(save_path, "span_curve"),
        #                 fig_id=idx, singlefile=True)
        #
        # MkdirCheck(os.path.join(save_path, "flow_std"))
        # plot_flow_std(post_pred, parameterList,
        #               work_path=os.path.join(save_path, "flow_std"),
        #               fig_id=idx, rangeIndex=50, singlefile=True)
        #
        # MkdirCheck(os.path.join(save_path, "flow_curve"))
        # plot_flow_curve(post_pred, parameterList,
        #                 work_path=os.path.join(save_path, "flow_curve"),
        #                 fig_id=idx, singlefile=True)
        pred = None

    for parameter in parameterList:
        fig = dict_fig[parameter]
        plt.figure(fig.number)
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        jpg_path = os.path.join(save_path, parameter + "_flow_" + "all_" + '.jpg')
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
        fig.savefig(jpg_path)
        plt.close(fig)

