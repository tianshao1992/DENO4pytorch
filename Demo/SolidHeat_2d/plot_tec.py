import os
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import time as record
#数据集加载
import GCN_data as Rdataset
import GCN_model as Model
import visual_data as Visual

train_size = 0.6
valid_size = 0.2
data_name = '5000ljx_12-6'
path = 'results_ljx\\' + data_name + "_test_U" + str(train_size) + "depth8_16-6"
if not os.path.exists(path):
    os.makedirs(path)


# transform = T.Cartesian(cat=False)
transform = T.Distance(cat=False)
dataset = Rdataset.PDataset(data_name, root=path, transform=transform, if_download=False)
design_norm = dataset.norms[0]
coords_norm = dataset.norms[1]
fields_norm = dataset.norms[2]

name='temp_0.5span_limitss_ljx'

all = torch.load('data\\GNN_' + name + '.pth', 'r')
design = all['design']
elements_ = all['elems']

del all
# 训练集，测试集划分 拉丁超立方打乱
np.random.seed(2021)
shuffle_number = np.random.permutation(len(dataset)).tolist()
dataset = dataset[shuffle_number]
elements = []

for shuffle_id in shuffle_number:
    elements.append(elements_[shuffle_id])
# dataset = dataset[2001:3000]
# elements = elements_
n = int(len(dataset) * train_size)

n2 = int(len(dataset) * (1-valid_size))

valid_nmber = int(len(dataset))-n


train_dataset = dataset[:n]
valid_dataset = dataset[n2:]
train_element = elements[:n]
valid_element = elements[n2:]
num_find = shuffle_number[n2:]

del dataset
train_loader = DataLoader(train_dataset, batch_size=8,  shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
visual_loader = DataLoader(valid_dataset[-50:], batch_size=1,  shuffle=False)
visual_element = valid_element[-50:]

num_find = num_find[-50:]



input_channels = 15 #12个设计变量+3个坐标
output_channels = 1 #输出温度场

planes = [input_channels,] + [32, 64, 128, 128, 128, 64, 32] + [output_channels]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model.SAGENet_U(planes).to(device)

recog = Model.Recognizer_SAGE_all([16, 6]).to(device)

model.load_state_dict(torch.load(os.path.join(path, 'latest_model.pth'))['model'])
recog.load_state_dict(torch.load(os.path.join(path, 'latest_recog.pth'))['recog'])

dynam = Model.Dynamicor(device)
optimizer = torch.optim.Adamax(model.parameters(), lr=0.005)
boundary_epoch = [201, 301, 401, 501]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=boundary_epoch, gamma=0.1)
# fields_loss = nn.MSELoss().to(device)
fields_loss = Model.BoundaryMSE(device=device, weights=1.)

visual = Visual.matplotlib_vision(path)
error = Model.prediction_error(device=device, nodes_num=336*5)

design_log = []
ele_itr_log = []


loader = iter(visual_loader)

for i in range(10):
    model.eval()
    with torch.no_grad():
        # for data in loader:
        data = next(loader).to(device)
        # data = loader[0]
        f = model(data)
        design = error.design_para(data.batch, data.x)
        design = design_norm.back(design)
    xy, r, f = data.x[:, 0:3].cpu().numpy(), data.y.cpu().numpy(), f.cpu().numpy()

    ele_itr = num_find[i]+1
    visual_element_ = visual_element[i]

    xy = coords_norm.back(xy)

    r = fields_norm.back(r)
    f = fields_norm.back(f)

    design_log.append(design)
    ele_itr_log.append(ele_itr)

    visual.plot_fields_ASC3(ele_itr, xy[:, 0], xy[:, 1], xy[:, 2], r, f, visual_element_)
    visual.plot_fields_CSV(ele_itr, xy[:, 0], xy[:, 1], xy[:, 2], r, f, r-f, visual_element_)

design_log = np.concatenate(design_log, axis=0)
ele_itr_log = np.array(ele_itr_log).reshape(-1, 1)

np.savetxt(path + '\\design.txt',  np.concatenate((ele_itr_log,design_log), axis=1))

