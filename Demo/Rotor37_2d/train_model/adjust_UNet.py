import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
print(torch.cuda.device_count())

from post_process.load_model import build_model_yml, loaddata
from model_whole_life import WorkPrj, DLModelWhole, change_yml, add_yml
from adjust_FNO import work_construct, work_construct_togethor


if __name__ == "__main__":
    name = "UNet"
    start_id = 5
    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    dict3 = {
        'modes': [6,8],
        'width': [64],
        'depth': [4],
        'dropput': [0],
        'activation': ['gelu'],
        'batch_size':[16,32,64]
    }

    # model_list1 = work_construct_togethor(dict1)
    # model_list2 = work_construct_togethor(dict2)
    model_list3 = work_construct_togethor(dict3)

    for id, config_dict in enumerate(model_list3):
        work = WorkPrj(os.path.join("..", "work_train_UNet", name + "_" + str(id + start_id)))

        change_yml(name, yml_path=work.yml, **config_dict)
        add_yml(["Optimizer_config", "Scheduler_config", "Basic_config"], yml_path=work.yml)
        train_loader, valid_loader, x_normalizer, y_normalizer = loaddata(name, **work.config("Basic"))

        x_normalizer.save(work.x_norm)
        y_normalizer.save(work.y_norm)
        DL_model = DLModelWhole(Device, name=name, work=work)
        DL_model.set_los()
        DL_model.train_epochs(train_loader, valid_loader)
