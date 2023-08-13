from draw_sensitive import *
from Utilizes.visual_data import MatplotlibVision

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
        # "PressureLossR",
        # "EntropyStatic",
        "MachIsentropic",
        # "Load",
    ]


    #设置分组

    # var_group = list(range(28))
    # var_group = [[x] for x in var_group]

    # 按叶高分组
    var_group = [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24],
                [25, 26, 27],
                ]

    #按流向位置分组
    # var_group = [
    #     [0, 5, 10, 15, 20],
    #     [1, 6, 11, 16, 21],
    #     [2, 7, 12, 17, 22],
    #     [3, 8, 13, 18, 23],
    #     [4, 9, 14, 19, 24],
    #     [25, 26, 27],
    # ]



    dict_fig = {}
    dict_axs = {}
    dict_data = {}
    for ii, parameter in enumerate(parameterList):
        fig, axs = plt.subplots(1, 1, figsize=(5, 5), num=ii)
        data = np.zeros([len(var_group)])
        dict_fig.update({parameter: fig})
        dict_axs.update({parameter: axs})
        dict_data.update({parameter: data})

    #开始计算数据

    for idx, var_list in enumerate(var_group):
        sample_grid = mesh_sliced(input_dim, var_list, sample_num=1001*len(var_list))
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

        for ii, parameter in enumerate(parameterList):
            #计算展向数据
            # value_span = getattr(post_pred, parameter)  # shape = [num, 64, 64]
            # value_span = np.mean(value_span[:, :, -5:], axis=2)  # shape = [num, 64]
            # normalizer = DataNormer(value_span, method='mean-std', axis=0)  # 这里对网格上的具体数据进行平均

            #计算流向曲线
            # value_flow = post_pred.field_density_average(parameter, location="whole")
            # normalizer = DataNormer(value_flow, method='mean-std', axis=(0,))

            #计算0维数据
            value_span = getattr(post_pred, parameter)
            value_span = np.mean(value_span[:, :, 15:17], axis=2)
            value_span = post_pred.span_density_average(value_span)
            normalizer = DataNormer(value_span, method='mean-std', axis=0)

            dict_data[parameter][idx] = normalizer.std # 将数据加入

    #数据计算结束


    # save_path = os.path.join(work_path, "sensitive_test")
    save_path = os.path.join("..", "data", "final_fig")
    Visual = MatplotlibVision(work_path, input_name=('Z', 'R'), field_name=('unset'))  # 不在此处设置名称

    for parameter in parameterList:
        fig = dict_fig[parameter]
        axs = dict_axs[parameter]
        data = dict_data[parameter]

        data = data/np.sum(data)

        plt.figure(fig.number)
        plt.subplots_adjust(hspace=0.05, wspace=0.05)

        Visual.plot_pie(fig, axs, data, title=None)

        jpg_path = os.path.join(save_path, parameter + "_pie_upper" + '.jpg')
        fig.tight_layout()
        fig.savefig(jpg_path)
        plt.close(fig)



