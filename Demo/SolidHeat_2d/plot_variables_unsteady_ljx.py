import os
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time as record
import h5py

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

input_channels = 15 #12个设计变量+3个坐标
output_channels = 1 #输出温度场

planes = [input_channels,] + [32, 64, 128, 128, 128, 64, 32] + [output_channels]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model.SAGENet_U(planes).to(device)
recog = Model.Recognizer_SAGE_all([16, 6]).to(device)

# 模型读入
checkpoint = torch.load(os.path.join(path, 'latest_model.pth'))
model.load_state_dict(checkpoint['model'])

checkpoint = torch.load(os.path.join(path, 'latest_recog.pth'))
recog.load_state_dict(checkpoint['recog'])


name='temp_0.5span_limitss_ljx'
all = torch.load('data\\GNN_' + name + '.pth', 'r')
design = all['design']
elements = all['elems']

valid_element = elements[:]
visual_element = valid_element[-50:]



# transform = T.Cartesian(cat=False)
transform = T.Distance(cat=False)
dataset = Rdataset.PDataset(data_name, root=path, transform=transform, if_download=False)
design_norm = dataset.norms[0]
coords_norm = dataset.norms[1]
fields_norm = dataset.norms[2]
target_norm = dataset.norms[3]
#
#all = torch.load('data\\GNN_data_10000' + '.pth', 'r')
# elements = all['elems']
# elements = elements[:1000]
#
# del all

# 训练集，测试集划分

np.random.seed(2021)


# dataset = dataset[:1000]
n = int(len(dataset) * train_size)
n2 = n + int(len(dataset) * valid_size)
train_dataset = dataset[:n]
valid_dataset = dataset[n:n2]
#
# train_element = elements[:n]
# valid_element = elements[n:]

#del dataset

train_loader = DataLoader(train_dataset, batch_size=16,  shuffle=True)
#valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, drop_last=False)
#visual_loader = DataLoader(valid_dataset[-50:], batch_size=1,  shuffle=False)
data = next(iter(valid_loader))
#data_visual = next(iter(visual_loader))
visual = Visual.matplotlib_vision(path)

#for i in (0, 1, 2, 6): #所有非定常参数
design_x = torch.zeros((20, 12), dtype=torch.float).cuda()
design_= torch.linspace(-np.pi, np.pi, 20).cuda()
temperature=0.05*torch.sin(design_)
pressure=0.15*torch.sin(design_)
mass=torch.sin(design_)
design_angle= torch.linspace(-1.5*np.pi, 0.5*np.pi, 20).cuda()
inlet_angle=0.6*torch.sin(design_angle)
design_x[:, 0] = temperature
design_x[:, 1] = pressure
# design_x[:, 2] = mass
design_x[:, 6] = inlet_angle
f_log = []
t_log = []
for j in range(20): #####一组输入数据对应到1445个温度场节点上
    data.x[:1445, 3:15] = design_x[j,:]
#        test=data.x.cpu().numpy()
    with torch.no_grad():
        data = data.to(device)
        f = model(data)
        t=recog(data,f)
        t = target_norm.back(t).cpu().numpy()

    xy, r, f = data.x[:, :3].cpu().numpy(), data.y.cpu().numpy(), f.cpu().numpy()

    ele_itr = j
    visual_element_ = visual_element[ele_itr]  ###假的网格

    xy = coords_norm.back(xy)

    r = fields_norm.back(r)
    f = fields_norm.back(f)

    t_log.append(t)
    f_log.append(f)
    visual.plot_fields_ASC3(ele_itr+10000, xy[:, 0], xy[:, 1], xy[:, 2], f, f, visual_element_)

####

visual.plot_fields_ASC3(0, xy[:, 0], xy[:, 1], xy[:, 2], f_log[0], f_log[15], visual_element_)
visual.plot_fields_ASC3(5, xy[:, 0], xy[:, 1], xy[:, 2], f_log[5], f_log[0], visual_element_)
visual.plot_fields_ASC3(10, xy[:, 0], xy[:, 1], xy[:, 2], f_log[10], f_log[5], visual_element_)
visual.plot_fields_ASC3(15, xy[:, 0], xy[:, 1], xy[:, 2], f_log[15], f_log[10], visual_element_)

design_x = design_norm.back(design_x.cpu().numpy())

t_log = np.concatenate(t_log, axis=0)
    # for task, ax in enumerate(axs):
    #     ax.plot(design_x[:, i], t[:, task], 'b')
    #     ax.set_title(f'Task {task + 1}')
    #     # Shade in confidence
    #     #ax.fill_between(design_x[:, i], lower[:, task], upper[:, task], alpha=0.5)
    #     #ax.legend(['Observed Data', 'Mean', 'Confidence'])
    # plt.savefig(path + '\\design-target.svg')
np.savetxt(path + '\\unsteady-design.csv', (design_x[:, 0], design_x[:, 1], design_x[:, 2], design_x[:, 6]),fmt='%.18e', delimiter=',')  # 输出非定常输入参数
np.savetxt(path + '\\unsteady-pre.csv', (t_log[:, 0],t_log[:, 1],t_log[:, 2],t_log[:, 3],t_log[:, 4],t_log[:, 5]), fmt='%.18e', delimiter=',') #输出性能参数预测值
#    np.savetxt(path + '\\location_' + str(i) + '-pre.csv', (design_x[:, i], t[:, 5], t[:, 6], t[:, 7], t[:, 9], t[:, 10], t[:, 11]), fmt='%.18e', delimiter=',')  # 输出位置预测值
