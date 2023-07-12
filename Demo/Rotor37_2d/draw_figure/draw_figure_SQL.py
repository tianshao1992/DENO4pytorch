from utilizes_draw import *
from post_process.load_model import loaddata_Sql

if __name__ == "__main__":

    name = 'Transformer_0'
    input_dim = 363
    output_dim = 5
    work_load_path = os.path.join("..", "work_trainsql_Trans1")
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
        train_loader, valid_loader, _, _ = loaddata_Sql(nameReal, 2500, 400, shuffled=True)

        for type in ["valid"]:
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
            # plot_error_box(true, pred, save_path=None, type=type, work_path=work_path)

            post_true = Post_2d(true, grid,
                                inputDict=input_para,
                                )
            post_pred = Post_2d(pred, grid,
                                inputDict=input_para,
                                )

            # parameterList = []
            # parameterList = [
            #                  "Efficiency", "EfficiencyPoly",
            #                  "PressureRatioV", "TemperatureRatioV",
            #                  "PressureLossR", "EntropyStatic",
            #                  "MachIsentropic",
            #                  "Load",
            #                  ]
            parameterList = [
                "PressureTotalV",
                "TemperatureTotalV",
                "EntropyStatic",
                "MachIsentropic",
                "Load"
            ]
            # parameterListN = [
            #     "PR", "TR",
            #     "Eff", "EffPoly",
            #     "PLoss", "Entr",
            #     "Mach", "Load",
            #     "MF"]

            # for kk, parameter_Name in enumerate(parameterList):
            #     true[...,kk] = getattr(post_true, parameter_Name)
            #     pred[...,kk] = getattr(post_pred, parameter_Name)

            # dict = plot_error(post_true, post_pred, parameterList + ["MassFlow"],
            #            paraNameList=parameterListN,
            #            save_path=None, fig_id=0, label=None, work_path=work_path, type=type)

            # if type=="valid":
            #     np.savez(os.path.join(work_path, "FNO_num.npz"), **dict)

            plot_field_2d(post_true, post_pred, parameterList, work_path=work_path, type=type, grid=grid)
            Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('p', 't', 'V', 'W', 'mass'))
            # field_name = ['p[kPa]', 't[K]', '$\mathrm{v^{2}/1000[m^{2}s^{-2}]}$', '$\mathrm{w^{2}/1000[m^{2}s^{-2}]}$', '$\mathrm{mass [kg m^{-2}s^{-1}]}$']
            field_name = ['p[kPa]', 't[K]',
                          '$\mathrm{S[J/K]}$',
                          '$\mathrm{Ma_{is}}$',
                          '$\mathrm{\psi}$']

            # scaleList = [1000, 1, 1, 1, 1]
            scaleList = [1,1,1,1,1]


            # limitList = [20,20,50,0.1,0.1]
            for ii in [1, 2, 3]:
                for jj in range(5):
                    true[ii, :, :, jj:jj + 1] = true[ii, :, :, jj:jj + 1]/scaleList[jj]
                    pred[ii, :, :, jj:jj + 1] = pred[ii, :, :, jj:jj + 1] / scaleList[jj]

                    # fig, axs = plt.subplots(3, 1, figsize=(30, 6), num=jj)
                    # Visual.field_name = [field_name[jj]]
                    # Visual.plot_fields_ms_col(fig, axs, (true[ii,:,:,jj:jj+1]), (pred[ii,:,:,jj:jj+1])
                    #                           , grid, show_channel=[0],cmaps = ['Spectral_r', 'Spectral_r', 'coolwarm'],
                    #                           limit=limitList[jj])
                # fig, axs = plt.subplots(5, 3, figsize=(25, 15), num=1)

                fig, axs = plt.subplots(5, 3, figsize=(45, 30), num=1)
                Visual.field_name = field_name
                Visual.plot_fields_ms(fig, axs, true[ii], pred[ii],
                                      grid, show_channel=None, cmaps=['Spectral_r', 'Spectral_r', 'coolwarm'],
                                      limitList=None)

                save_path = work_path
                # fig.patch.set_alpha(0.)
                fig.savefig(os.path.join(save_path, 'derive' + str(0) + '_field_' + str(ii) + '.png'),transparent=True)
                plt.close(fig)




                # post_compare = Post_2d(np.concatenate((true[ii:ii + 1, :], pred[ii:ii + 1, :]), axis=0), grid,
                #                        inputDict=input_para,
                #                        )
                # plot_span_curve(post_compare, parameterList,
                #                 save_path=None, fig_id=ii, label=None, type=type, work_path=work_path)
