import numpy as np

from utilizes_draw import *


if __name__ == "__main__":

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
    ErrBox = np.zeros([5,6])
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


        nameReal = name.split("_")[0]
        id = None
        if len(name.split("_")) == 2:
            id = int(name.split("_")[1])

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
        train_loader, valid_loader, _, _ = loaddata(nameReal, 2500, 400, shuffled=True)

        for type in ["valid"]:
            if type == "valid":
                true, pred = get_true_pred(valid_loader, Net_model, inference, Device,
                                           name=nameReal, iters=10, alldata=True)
            elif type == "train":
                true, pred = get_true_pred(train_loader, Net_model, inference, Device,
                                           name=nameReal, iters=10, alldata=True)

            true = y_normalizer.back(true)
            pred = y_normalizer.back(pred)

            Error_func = FieldsLpLoss(size_average=False)
            Error_func.p = 2
            ErrL2r = Error_func.rel(pred, true)
            ErrL2r = np.mean(ErrL2r, axis=0)

            ErrBox[ii, :5] = ErrL2r.copy()
            ErrBox[ii,5] = np.mean(ErrL2r, axis=0)

    save_path = os.path.join("..", "data", "final_fig",'data.csv')
    np.savetxt(save_path, ErrBox, delimiter=',')
