#2023-4-24
from Utilizes.visual_data import MatplotlibVision
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from Utilizes.loss_metrics import FieldsLpLoss


if __name__ == "__main__":
    # name = 'MLP'
    filenameList =[
     'work',
     # 'work_2700_MSELoss',
     # 'work_FNO_mode_L1smoothLoss',
     # 'work_FNO_mode_MSELoss',
     # 'work_L1smoothLoss',
     # 'work_MSELoss',
    ]

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')


    for filename in filenameList:
        work_load_path = os.path.join(r"D:\WQN\CODE\DENO4pytorch-main\Demo\Rotor37_2d/",filename)
        workList = os.listdir(work_load_path)
        for name in workList:#['MLP','deepONet','FNO','UNet','Transformer']:
            nameReal = name.split("_")[0]
            modes = 10
            if len(name.split("_"))==2:
                modes = int(name.split("_")[1])
            work_path = os.path.join(work_load_path,name)

            # load_data
            if os.path.exists(os.path.join(work_path, "true_train.npy")):
                train_true = torch.tensor(np.load(os.path.join(work_path, "true_train.npy")), dtype=torch.float)
                valid_true = torch.tensor(np.load(os.path.join(work_path, "true_valid.npy")), dtype=torch.float)
                train_pred = torch.tensor(np.load(os.path.join(work_path, "pred_train.npy")), dtype=torch.float)
                valid_pred = torch.tensor(np.load(os.path.join(work_path, "pred_valid.npy")), dtype=torch.float)

                Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('p', 't', 'rho', 'Vx', 'Vy'))
                Loss_func = nn.MSELoss()
                Error_func = FieldsLpLoss(size_average=False)

                Error_func.p = 1
                ErrL1a = Error_func.abs(valid_pred, valid_true)
                ErrL1r = Error_func.rel(valid_pred, valid_true)
                Error_func.p = 2
                ErrL2a = Error_func.abs(valid_pred, valid_true)
                ErrL2r = Error_func.rel(valid_pred, valid_true)

                fig, axs = plt.subplots(1, 1, figsize=(10, 10), num=1)
                Visual.plot_box(fig, axs, ErrL2r, xticks=Visual.field_name)
                fig.savefig(os.path.join(work_path, 'valid_box.jpg'))

                plt.close(fig)

            # finally:
            #     print(filename+"_"+name)
            #     print("error")
