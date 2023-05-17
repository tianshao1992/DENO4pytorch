import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
torch.cuda.set_device(0)
from post_process.load_model import build_model_yml, loaddata
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
    name = "FNO"
    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')
    start_id = 9
    dict = {
    'modes': [4, 6],
    'width': [128],
    'depth': [4, 6],
    'activation': ['gelu', 'relu']
    }

    worklist = work_construct_togethor(dict)

    for id, config_dict in enumerate(worklist):
        work = WorkPrj(os.path.join("..", "work_train_FNO2", name + "_" + str(id)))
        change_yml(name, yml_path=work.yml, **config_dict)
        add_yml(["Optimizer_config", "Scheduler_config", "Basic_config"], yml_path=work.yml)
        train_loader, valid_loader, x_normalizer, y_normalizer = loaddata(name, **work.config("Basic"))
        x_normalizer.save(work.x_norm)
        y_normalizer.save(work.y_norm)
        DL_model = DLModelWhole(Device, name=name, work=work)
        DL_model.set_los()
        DL_model.train_epochs(train_loader, valid_loader)


