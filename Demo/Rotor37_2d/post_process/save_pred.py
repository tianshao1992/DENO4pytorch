#2023-4-24
#statistics of results
import torch
import os
import numpy as np
from post_data import Post_2d
from run_MLP import get_grid, get_origin

from load_model import loaddata, rebuild_model
import yaml

def get_true_pred(loader, Net_model, Device):
    if 'MLP' in name:
        grid, true, pred = inference(loader, Net_model, Device)
    else:
        coord, grid, true, pred = inference(loader, Net_model, Device)
    true = true.reshape([true.shape[0], 64, 64, out_dim])
    pred = pred.reshape([pred.shape[0], 64, 64, out_dim])

    return true, pred



if __name__ == "__main__":
    # name = 'MLP'
    filenameList =[
        "work",
     # 'work_2700_MSELoss',
     # 'work_FNO_mode_L1smoothLoss',
     # 'work_FNO_mode_MSELoss',
     # 'work_L1smoothLoss',
     # 'work_MSELoss',
    ]

    for filename in filenameList:
        work_load_path = os.path.join(r"D:\WQN\CODE\DENO4pytorch-main\Demo\Rotor37_2d/",filename)
        workList = os.listdir(work_load_path)
        out_dim = 5
        for name in workList:#['MLP','deepONet','FNO','UNet','Transformer']:
            nameReal = name.split("_")[0]
            mode = 10
            if len(name.split("_"))==2:
                mode = int(name.split("_")[1])
            work_path = os.path.join(work_load_path,name)
            if torch.cuda.is_available():
                Device = torch.device('cuda')
            else:
                Device = torch.device('cpu')

            Net_model, inference = rebuild_model(work_path, Device)
            if Net_model is not None:
                # load data
                train_loader, valid_loader, x_normalizer, y_normalizer = loaddata(nameReal,1250,150)

                train_true, train_pred = get_true_pred(train_loader, Net_model, Device)
                valid_true, valid_pred = get_true_pred(valid_loader, Net_model, Device)

                np.save(os.path.join(work_path, "true_train.npy"), train_true)
                np.save(os.path.join(work_path, "true_valid.npy"), valid_true)
                np.save(os.path.join(work_path, "pred_train.npy"), train_pred)
                np.save(os.path.join(work_path, "pred_valid.npy"), valid_pred)








