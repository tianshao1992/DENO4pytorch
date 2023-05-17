import os
import torch
import numpy as np
from post_process.post_data import Post_2d
from run_FNO import feature_transform
from Demo.Rotor37_2d.utilizes_rotor37 import get_grid
from utilizes_rotor37 import Rotor37WeightLoss
from post_process.load_model import build_model_yml, loaddata
from post_process.model_predict import DLModelPost
from Utilizes.visual_data import MatplotlibVision
import matplotlib.pyplot as plt
import yaml
import time


def change_yml(name, yml_path=None, **kwargs):
    # 加载config模板
    template_path = os.path.join("..", "data", "config_template.yml")
    with open(template_path) as f:
        config_all = yaml.full_load(f)
        config_para = config_all[name + '_config']
    # 修改参数
    for key in kwargs.keys():
        if key in config_para.keys():
            config_para[key] = kwargs[key]
        else:
            print("The keywords {} is illegal, CHECK PLEASE".format(key))
    # 保存到新的文件
    isExist = os.path.exists(yml_path)
    if not isExist:
        with open(yml_path, 'w') as f:
            pass
    with open(yml_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if data is None:
        data = {}
    data[name + '_config'] = config_para
    with open(yml_path, 'w') as f:
        yaml.dump(data, f)

def add_yml(key_set_list, yml_path=None):
    template_path = os.path.join("..", "data", "config_template.yml")
    with open(template_path) as f:
    # 加载config模板
        config_all = yaml.full_load(f)
    for key_set in key_set_list:
        config_para = config_all[key_set]
    # 保存到新的文件
        with open(yml_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        data[key_set] = config_para
        with open(yml_path, 'w') as f:
            yaml.dump(data, f)

class WorkPrj(object):
    def __init__(self, work_path):
        self.root = work_path
        isExist = os.path.exists(self.root)
        if not isExist:
            os.mkdir(self.root)
        self.pth = os.path.join(self.root, 'latest_model.pth')
        self.svg = os.path.join(self.root, 'log_loss.svg')
        self.yml= os.path.join(self.root, 'config.yml')
        self.x_norm = os.path.join(self.root, 'x_norm.pkl')
        self.y_norm = os.path.join(self.root, 'y_norm.pkl')

    def config(self, name):
        with open(self.yml) as f:
            config_all = yaml.full_load(f)

        if name + "_config" in config_all.keys():
            return config_all[name + "_config"]
        else:
            return None



    # def exist_check(self):

class DLModelWhole(object):
    def __init__(self, device,
                 name=None,
                 in_norm=None,
                 out_norm=None,
                 grid_size=64,
                 work=None,
                 epochs=1000,
                 ):
        self.device = device
        self.work = work
        self.name = name
        self.net_model, self.inference, self.train, self.valid = \
            build_model_yml(work.yml, self.device, name=name)

        self.in_norm = in_norm
        self.out_norm = out_norm
        self.grid_size = grid_size

        self.epochs = epochs

        self.Loss_func = None
        self.Optimizer = None
        self.Scheduler = None

    def set_los(self):
        with open(self.work.yml) as f:
            config = yaml.full_load(f)
        # 损失函数
        # self.Loss_func = torch.nn.MSELoss()
        self.Loss_func = Rotor37WeightLoss()
        # 优化算法
        temp = config["Optimizer_config"]
        temp['betas'] = tuple(float(x) for x in temp['betas'][0].split())
        self.Optimizer = torch.optim.Adam(self.net_model.parameters(), **temp)
        # 下降策略
        self.Scheduler = torch.optim.lr_scheduler.StepLR(self.Optimizer, **config["Scheduler_config"])

    def train_epochs(self, train_loader, valid_loader):
        work = self.work
        Visual = MatplotlibVision(work.root, input_name=('x', 'y'), field_name=('unset',))
        star_time = time.time()
        log_loss = [[], []]
        for epoch in range(self.epochs):
            self.net_model.train()
            log_loss[0].append(self.train(train_loader, self.net_model, self.device, self.Loss_func, self.Optimizer, self.Scheduler))
            self.net_model.eval()
            log_loss[1].append(self.valid(valid_loader,self.net_model, self.device, self.Loss_func))
            print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
                  format(epoch, self.Optimizer.param_groups[0]['lr'], log_loss[0][-1], log_loss[1][-1],
                         time.time() - star_time))
            # print(os.environ['CUDA_VISIBLE_DEVICES'])
            star_time = time.time()

            if epoch > 0 and epoch % 5 == 0:
                fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
                Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
                Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
                fig.suptitle('training loss')
                fig.savefig(work.svg)
                plt.close(fig)

            if epoch > 0 and epoch % 100 == 0:
                torch.save(
                    {'log_loss': log_loss, 'net_model': self.net_model.state_dict(), 'optimizer': self.Optimizer.state_dict()},
                    work.pth)


if __name__ == "__main__":
    name = "MLP"
    id = 0
    train_num = 2500
    valid_num = 450
    work = WorkPrj(os.path.join("..", "work_train", name + "_" + str(id)))

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    config_dict = {
                    'n_hidden': 512,
                    'num_layers': 10,
                  }
    change_yml(name, yml_path=work.yml, **config_dict)
    add_yml(["Optimizer_config", "Scheduler_config"], yml_path=work.yml)
    train_loader, valid_loader, x_normalizer, y_normalizer = loaddata(name, train_num, valid_num, shuffled=True)
    x_normalizer.save(work.x_norm)
    y_normalizer.save(work.y_norm)
    DL_model = DLModelWhole(Device, name=name, work=work)
    DL_model.set_los()
    DL_model.train_epochs(train_loader, valid_loader)

    post = DLModelPost(DL_model.net_model, Device, name=name, in_norm=x_normalizer, out_norm=y_normalizer)

    Rst = []
    for batch, (input, output) in enumerate(valid_loader):
        Rst.append(post.predictor_value(input, parameterList="PressureLossR", input_norm=True))

    Rst = np.concatenate(Rst, axis=1)





