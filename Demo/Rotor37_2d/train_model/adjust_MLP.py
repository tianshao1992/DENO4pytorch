import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
torch.cuda.set_device(0)
# print(torch.cuda.device_count())
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from post_process.post_data import Post_2d
from run_FNO import feature_transform
from Demo.Rotor37_2d.utilizes_rotor37 import get_grid
from post_process.load_model import build_model_yml, loaddata
from post_process.model_predict import DLModelPost
from Utilizes.visual_data import MatplotlibVision
import matplotlib.pyplot as plt
import yaml
import time
from model_whole_life import WorkPrj, DLModelWhole, change_yml, add_yml

def work_construct(para_list_dict):
    work_list = []
    for key in para_list_dict.keys():
        num = len(para_list_dict[key])
        for ii in range(num):
            work_list.append({key:para_list_dict[key][ii]})

    return work_list

def work_construct_togethor(para_list_dict):
    work_list = []
    num = 1
    for key in para_list_dict.keys():
        num = num * len(para_list_dict[key])

    for ii in range(num):
        dict_new = {}
        for key in para_list_dict.keys():
            idx = ii % len(para_list_dict[key])
            dict_new.update({key:para_list_dict[key][idx]})
        work_list.append(dict_new)
    return work_list



if __name__ == "__main__":
    name = "MLP"
    start_id = 0
    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')
    dict_model = {
                "n_hidden": [256, 512],
                "num_layers": [8, 12, 14],
                "is_BatchNorm": [True, False]
                }

    model_list = work_construct(dict_model)

    for id, config_dict in enumerate(model_list):
        work = WorkPrj(os.path.join("..", "work_train_MLP", name + "_" + str(id + start_id)))

        change_yml(name, yml_path=work.yml, **config_dict)
        add_yml(["Optimizer_config", "Scheduler_config", "Basic_config"], yml_path=work.yml)

        train_loader, valid_loader, x_normalizer, y_normalizer = loaddata(name, **work.config("Basic"))
        x_normalizer.save(work.x_norm)
        y_normalizer.save(work.y_norm)
        DL_model = DLModelWhole(Device, name=name, work=work)
        DL_model.set_los()
        DL_model.train_epochs(train_loader, valid_loader)
