#2023-4-24
#statistics of results
import torch
import os
import numpy as np
from post_data import Post_2d
from run_MLP import get_grid, get_origin, valid
from train_model.model_whole_life import WorkPrj
from load_model import loaddata, rebuild_model, get_true_pred, build_model_yml
import yaml

if __name__ == "__main__":
    # name = 'MLP'
    filenameList =[
        "work_train",
     # 'work_2700_MSELoss',
     # 'work_FNO_mode_L1smoothLoss',
     # 'work_FNO_mode_MSELoss',
     # 'work_L1smoothLoss',
     # 'work_MSELoss',
    ]

    for filename in filenameList:
        work_load_path = os.path.join("..", filename)
        workList = os.listdir(work_load_path)
        out_dim = 5
        for name in ['FNO_0']:#workList:#['MLP','deepONet','FNO','UNet','Transformer']:
            nameReal = name.split("_")[0]
            mode = None
            if len(name.split("_"))==2:
                mode = int(name.split("_")[1])

            work_path = os.path.join(work_load_path,name)
            work = WorkPrj(work_path)

            if torch.cuda.is_available():
                Device = torch.device('cuda')
            else:
                Device = torch.device('cpu')

            if os.path.exists(work.yml):
                Net_model, inference, _, _ = build_model_yml(work.yml, Device, name=nameReal)
                isExist = os.path.exists(work.pth)
                if isExist:
                    checkpoint = torch.load(work.pth, map_location=Device)
                    Net_model.load_state_dict(checkpoint['net_model'])
            else:
                if mode is not None:
                    Net_model, inference = rebuild_model(work_path, Device, name=nameReal, mode=mode)
                else:
                    Net_model, inference = rebuild_model(work_path, Device, name=nameReal)
            Net_model.eval()
            
            if Net_model is not None:
                # load data
                train_loader, valid_loader, x_normalizer, y_normalizer = loaddata(nameReal, 2500, 400, shuffled=True)

                train_true, train_pred = get_true_pred(train_loader, Net_model, inference, Device, name=nameReal)
                valid_true, valid_pred = get_true_pred(valid_loader, Net_model, inference, Device, name=nameReal)

                np.save(os.path.join(work_path, "true_train.npy"), train_true)
                np.save(os.path.join(work_path, "true_valid.npy"), valid_true)
                np.save(os.path.join(work_path, "pred_train.npy"), train_pred)
                np.save(os.path.join(work_path, "pred_valid.npy"), valid_pred)








