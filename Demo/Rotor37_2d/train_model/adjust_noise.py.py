import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
print(torch.cuda.device_count())
from post_process.load_model import build_model_yml, loaddata
from model_whole_life import WorkPrj, DLModelWhole, change_yml, add_yml

def work_construct(para_list_dict):
    work_list = []
    for key in para_list_dict.keys():
        num = len(para_list_dict[key])
        for ii in range(num):
            work_list.append({key:para_list_dict[key][ii]})

    return work_list



if __name__ == "__main__":
    name = "UNet"
    start_id = 0
    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    dict = {
        'ntrain': [500, 1000, 1500, 2000],
        'noise_scale': [0.005, 0.01, 0.05, 0.1],
    }

    worklist = work_construct(dict)

    for id, config_dict in enumerate(worklist):
        work = WorkPrj(os.path.join("..", "work_noise_UNet", name + "_" + str(id + start_id)))
        change_yml("Basic", yml_path=work.yml, **config_dict)
        add_yml(["Optimizer_config", "Scheduler_config", name+"_config"], yml_path=work.yml)
        train_loader, valid_loader, x_normalizer, y_normalizer = loaddata(name, **work.config("Basic"))
        x_normalizer.save(work.x_norm)
        y_normalizer.save(work.y_norm)
        DL_model = DLModelWhole(Device, name=name, work=work)
        DL_model.set_los()
        DL_model.train_epochs(train_loader, valid_loader)


