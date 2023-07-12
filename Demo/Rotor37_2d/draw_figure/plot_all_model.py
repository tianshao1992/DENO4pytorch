import numpy as np
import matplotlib.pyplot as plt
import os
from Utilizes.visual_data import MatplotlibVision
from train_model.model_whole_life import WorkPrj
import torch
from utilizes_draw import plot_span_curve_marker, plot_error_new
from utilizes_rotor37 import get_grid, get_origin
from post_process.load_model import loaddata, rebuild_model, build_model_yml,get_true_pred
from Utilizes.process_data import DataNormer
# from draw_figure_pred import predicter
from post_process.post_data import Post_2d

def plot_loss():
    nameList = [
        # 'MLP',
        # 'UNet',
        # 'deepONet',
        # 'FNO',
        'Trans',
    ]
    scaleList = [1, 1, 1, 1, 1]
    pathList = [
        # 'work_train_MLP/MLP_5',
        # 'work_train_UNet/UNet_4',
        # 'work_train_deepONet/deepONet_1',
        # 'work_train_FNO2/FNO_1',
        # 'work_train_Trans2/Transformer_1',
        'work/Trunk_TRA',
    ]

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    fig, axs = plt.subplots(1, 1, figsize=(20, 8), num=1)
    colors = plt.cm.get_cmap('Dark2').colors[:5]

    for ii, path in enumerate(pathList):
        # name = 'FNO_1'
        input_dim = 28
        output_dim = 5
        work_path = os.path.join("..", path)

        # work_load_path = os.path.join("..", "work_train_FNO2")
        # work_path = os.path.join(work_load_path, name)
        work = WorkPrj(work_path)
        # torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
        #            os.path.join(work_path, 'latest_model.pth'))
        checkpoint = torch.load(os.path.join(work_path, 'latest_model.pth'), map_location=Device)
        log_loss = checkpoint['log_loss']
        Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('unset'))
        typeList = ['train','valid']
        rangelist = [0,1]
        colors = ['k',plt.cm.get_cmap('Dark2').colors[4]]
        for kk in [0,1]:
            num = 10
            loss_box = np.zeros([num,len(log_loss[1])])
            loss_box[0] = log_loss[kk]
            for jj in range(1,num):
                loss_box[jj, jj:] = log_loss[kk][:-jj]
                #buzu
                loss_box[jj,:jj] = np.tile(log_loss[kk][0],[1,jj])

            normalizer = DataNormer(loss_box, method='mean-std', axis=0)
            normalizer.std = np.clip(normalizer.std, 0, normalizer.std[1])
            Visual.plot_value_std_clean(fig, axs, np.arange(len(log_loss[kk])), normalizer.mean,
                                        # label=nameList[ii],color=colors[ii],
                                        label=typeList[kk],color=colors[kk],
                                        std=normalizer.std, stdaxis=1, rangeIndex=rangelist[kk],
                                        title=None, xylabels=("epoch", "loss value"),
                                        )



        # Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :] * scaleList[ii],
        #                  'train_' + nameList[ii], color=colors[ii], linestyle='--')
        # Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :] * scaleList[ii],
        #                  'valid_' + nameList[ii], color=colors[ii], linestyle='-')
        # fig.suptitle('training loss')

    # axs.legend(loc="best", ncol=2)
    # Visual.font["size"] = 11
    axs.set_xlim(0, 900)
    axs.set_ylim(1e-5, 1e-2)
    axs.legend(loc="best", framealpha=1, prop=Visual.font)
    save_path = os.path.join("..", "data", "final_fig")
    fig.savefig(os.path.join(save_path, 'log_loss_std_trans2_.jpg'))
    # plt.show()
    plt.close(fig)

def plot_1d_curves():
    nameList = [
        'MLP',
        'UNet',
        'deepONet',
        'FNO',
        'Transformer',
    ]
    scaleList = [1, 1, 1, 1, 1]
    pathList = [
        'work_train_MLP/MLP_5',
        'work_train_UNet/UNet_4',
        'work_train_deepONet/deepONet_1',
        'work_train_FNO2/FNO_1',
        # 'work_train_Trans2/Transformer_1',
        'work/Trunk_TRA',
    ]

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    # fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
    colors = plt.cm.get_cmap('tab10').colors[:5]

    # 获取真实样本
    grid = get_grid(real_path=os.path.join("..", "data"))
    # design, field = get_origin(realpath=os.path.join("..", "data"), getridbad=True, shuffled=True)
    # true = field[sample_number:sample_number+1, :, :, :]  # 只取一个样本

    predList = np.zeros([5, 64, 64, 64, 5])
    for ii, path in enumerate(pathList):
        name = nameList[ii]
        nameReal = name.split("_")[0]

        input_dim = 28
        output_dim = 5
        work_path = os.path.join("..", path)

        # work_load_path = os.path.join("..", "work_train_FNO2")
        # work_path = os.path.join(work_load_path, name)
        work = WorkPrj(work_path)

        id = None
        if len(name.split("_")) == 2:
            id = int(name.split("_")[1])

        if torch.cuda.is_available():
            Device = torch.device('cuda')
        else:
            Device = torch.device('cpu')

        norm_save_x = work.x_norm
        norm_save_y = work.y_norm

        x_normlizer = DataNormer(np.ndarray([1, 1]), method="mean-std", axis=0)
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

        train_loader, valid_loader, _, _ = loaddata(nameReal, 2500, 400, shuffled=True)
        true, pred = get_true_pred(valid_loader, Net_model, inference, Device,
                                   name=nameReal, iters=1, alldata=False)

        y_normalizer = DataNormer(np.ndarray([1, 1]), method="mean-std", axis=0)
        y_normalizer.load(norm_save_y)

        true = y_normalizer.back(true)
        pred = y_normalizer.back(pred)

        # pred = predicter(Net_model, torch.tensor(design[sample_number:sample_number+1,:]), Device, name=nameReal)  # 获得预测值

        # pred = pred.reshape([pred.shape[0], 64, 64, output_dim])
        # pred = y_normlizer.back(pred)
        # # name = 'FNO_1'
        predList[ii, :, :, :, :] = pred[:, :, :, :]

    trueList = true
    for sample_number in [5, 8, 58]:
        true = trueList[sample_number:sample_number + 1, :, :, :]
        pred = predList[:, sample_number, :, :, :]

        # Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('unset'))
        input_para = {
            "PressureStatic": 0,
            "TemperatureStatic": 1,
            "V2": 2,
            "W2": 3,
            "DensityFlow": 4,
        }
        ii = 0
        data = np.concatenate((true,pred), axis=0)
        post = Post_2d(data, grid,
                       inputDict=input_para,
                       )
        parameterList = [
            "Efficiency",
            # "EfficiencyPoly",
            "PressureRatioV",
            # "TemperatureRatioV",
            # "PressureLossR",
            "EntropyStatic",
            # "MachIsentropic",
            # "Load",
        ]

        colorlist = ['k']
        for jj in range(5):
            colorlist.append(colors[jj])
        # colorlist.append('k')

        # for parameter in parameterList:
        #     save_path = os.path.join("..", "data", "final_fig")
        #     plot_span_curve_marker(post, parameter,
        #                            singlefigure=False,
        #                            colorList=colorlist,
        #                            markerList=['-', '-', '-', '-', '-', '^'],
        #                            save_path=None, work_path=save_path, fig_id=sample_number,
        #                            label=None, type='',
        #                            singlefile=False)
        xlimList = [
            [0.6,0.98],
            [1.75,2.2],
            [0,115],
        ]
        labelList = [
            'CFD',
            'MLP',
            'UNet',
            'deepONet',
            'FNO',
            'Trans',
        ]
        for tt, parameter_Name in enumerate(parameterList):

            Visual = MatplotlibVision(work_path, input_name=('Z', 'R'), field_name=('n'))  # 不在此处设置名称
            fig, axs = plt.subplots(1, 1, figsize=(7, 9), num=1)
            value_span = getattr(post, parameter_Name)

            # for ii in range(post.num):
            #     Visual.plot_value(fig, axs, value_span[ii, :, -1], np.linspace(0, 1, post.n_1d), label=label,
            #                       title=parameter_Name, xylabels=(parameter_Name, "span"))

            Visual.plot_curve_scatter(fig, axs, value_span[:, ::2, -1],
                                      np.tile(np.linspace(0, 1, post.n_1d/2), (value_span.shape[0], 1)), labelList=labelList,
                                      colorList=colorlist, markerList=['_', '.', '.', '.', '.', '^'],
                                      title=parameter_Name, xylabels=("", ""),
                                      xlim=xlimList[tt])

            # axs.legend(loc="best", framealpha=0, prop=Visual.font)
            save_path = os.path.join("..", "data", "final_fig")
            plt.figure(dpi=1500)
            plt.tight_layout()
            fig.savefig(os.path.join(save_path, str(sample_number) + "_" + parameter_Name + 'curves.png'))
            # plt.show()
            plt.close(fig)


def plot_error():
    nameList = [
        'MLP',
        'UNet',
        'deepONet',
        'FNO',
        'Transformer',
    ]
    scaleList = [1, 1, 1, 1, 1]
    pathList = [
        'work_train_MLP/MLP_5',
        'work_train_UNet/UNet_4',
        'work_train_deepONet/deepONet_1',
        'work_train_FNO2/FNO_1',
        # 'work_train_Trans2/Transformer_1',
        'work/Trunk_TRA',
    ]

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    # fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
    colors = plt.cm.get_cmap('tab10').colors[:5]

    # 获取真实样本
    grid = get_grid(real_path=os.path.join("..", "data"))
    # design, field = get_origin(realpath=os.path.join("..", "data"), getridbad=True, shuffled=True)
    # true = field[sample_number:sample_number+1, :, :, :]  # 只取一个样本

    predList = []
    for ii, path in enumerate(pathList):
        name = nameList[ii]
        nameReal = name.split("_")[0]

        input_dim = 28
        output_dim = 5
        work_path = os.path.join("..", path)

        # work_load_path = os.path.join("..", "work_train_FNO2")
        # work_path = os.path.join(work_load_path, name)
        work = WorkPrj(work_path)

        id = None
        if len(name.split("_")) == 2:
            id = int(name.split("_")[1])

        if torch.cuda.is_available():
            Device = torch.device('cuda')
        else:
            Device = torch.device('cpu')

        norm_save_x = work.x_norm
        norm_save_y = work.y_norm

        x_normalizer = DataNormer(np.ndarray([1, 1]), method="mean-std", axis=0)
        x_normalizer.load(norm_save_x)
        y_normalizer = DataNormer(np.ndarray([1, 1]), method="mean-std", axis=0)
        y_normalizer.load(norm_save_y)

        if os.path.exists(work.yml):
            Net_model, inference, _, _ = build_model_yml(work.yml, Device, name=nameReal)
            isExist = os.path.exists(work.pth)
            if isExist:
                checkpoint = torch.load(work.pth, map_location=Device)
                Net_model.load_state_dict(checkpoint['net_model'])
        else:
            Net_model, inference = rebuild_model(work_path, Device, name=nameReal)

        Net_model.eval()

        train_loader, valid_loader, x_norm, y_norm = loaddata(nameReal, 2500, 400,
                                                              x_norm=x_normalizer,
                                                              y_norm=y_normalizer,
                                                              shuffled=True)
        true, pred = get_true_pred(valid_loader, Net_model, inference, Device,
                                   name=nameReal, iters=5, alldata=False
                                   )



        true = y_normalizer.back(true)
        pred = y_normalizer.back(pred)


        # if ii==1:
        #     true, pred = get_true_pred(valid_loader, Net_model, inference, Device,
        #                                name=nameReal, iters=5, alldata=False
        #                                )
        #
        #     true = y_normalizer.back(true)
        # else:
        #     _, pred = get_true_pred(valid_loader, Net_model, inference, Device,
        #                                name=nameReal, iters=5, alldata=False
        #                                )


        # pred = predicter(Net_model, torch.tensor(design[sample_number:sample_number+1,:]), Device, name=nameReal)  # 获得预测值

        # pred = pred.reshape([pred.shape[0], 64, 64, output_dim])
        # pred = y_normlizer.back(pred)
        # # name = 'FNO_1'
        predList.append(pred)

        # Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('unset'))
    input_para = {
        "PressureStatic": 0,
        "TemperatureStatic": 1,
        "V2": 2,
        "W2": 3,
        "DensityFlow": 4,
    }
    ii = 0
    data = np.concatenate((pred, true), axis=0)
    post_true = Post_2d(true, grid,
                        inputDict=input_para,
                        )
    post_pred_list = []

    for jj in range(len(pathList)):
        post_pred_list.append(Post_2d(predList[jj], grid,
                                      inputDict=input_para,
                                      ))
    parameterList = [
        "Efficiency",
        "EfficiencyPoly",
        "PressureRatioV",
        "TemperatureRatioV",
        "PressureLossR",
        "EntropyStatic",
        "MachIsentropic",
        "Load",
        "MassFlow"
    ]
    for parameter in parameterList:
        save_path = os.path.join("..", "data", "final_fig")

        plot_error_new(post_true, post_pred_list, parameter,
                       save_path=None, fig_id=0,
                       label=None, work_path=save_path, type=None, paraNameList=None,
                       colorList=colors
                       )

if __name__ == "__main__":
    # plot_loss()
    plot_1d_curves()
    # plot_error()



