import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
print(torch.cuda.device_count())
torch.cuda.set_device(0)
from post_process.load_model import loaddata
from model_whole_life import WorkPrj, DLModelWhole, change_yml, add_yml

def work_construct(para_list_dict):
    work_list = []
    for key in para_list_dict.keys():
        num = len(para_list_dict[key])
        for ii in range(num):
            work_list.append({key:para_list_dict[key][ii]})

    return work_list



if __name__ == "__main__":
    name = "Transformer"
    batch_size = 32
    start_id = 0
    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')
    train_num = 2500
    valid_num = 400
    dict_model = {
                'num_encoder_layers': [3, 5, 6],
                'n_hidden': [64, 128, 256],
                'dropout': [0.5],
                # 'encoder_dropout': [0.5],
                # 'decoder_dropout': [0.5],
                # 'n_head': [4,5,6],
                # 'activation': ['relu'],
    }

    model_list = work_construct(dict_model)

    for id, config_dict in enumerate(model_list):
        work = WorkPrj(os.path.join("..", "work_train_Trans", name + "_" + str(id + start_id)))

        change_yml(name, yml_path=work.yml, **config_dict)
        add_yml(["Optimizer_config", "Scheduler_config"], yml_path=work.yml)

        train_loader, valid_loader, x_normalizer, y_normalizer = loaddata(name, train_num, valid_num, shuffled=True)
        x_normalizer.save(work.x_norm)
        y_normalizer.save(work.y_norm)
        DL_model = DLModelWhole(Device, name=name, work=work)
        DL_model.set_los()
        DL_model.train_epochs(train_loader, valid_loader)
