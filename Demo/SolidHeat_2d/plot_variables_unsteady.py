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

train_size = 0.8
valid_size = 0.2
data_name = '10000'
path = 'results\\' + data_name + "_SAGEnet_" + str(train_size)
if not os.path.exists(path):
    os.makedirs(path)

input_channels = 15 #12个设计变量+3个坐标
output_channels = 1 #输出温度场

planes = [input_channels,] + [64, 128, 128, 128, 64] + [output_channels]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model.SAGENet(planes).to(device)
recog = Model.Recognizer([4, 12]).to(device) #输出功率、效率、最大温度、平均温度、最大应力及位置、最大应变及位置

# 模型读入
checkpoint = torch.load(os.path.join(path, 'latest_model.pth'))
model.load_state_dict(checkpoint['model'])

checkpoint = torch.load(os.path.join(path, 'latest_recog.pth'))
recog.load_state_dict(checkpoint['recog'])



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
valid_loader = DataLoader(valid_dataset, batch_size=20, shuffle=False, drop_last=False)
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
design_x[:, 2] = mass
design_x[:, 6] = inlet_angle
for j in range(20): #一组输入数据对应到1445个温度场节点上
    data.x[j*1445:((j+1)*1445), 3:15] = design_x[j,:]
#        test=data.x.cpu().numpy()
with torch.no_grad():
    data = data.to(device)
    # start_time = record.time()
    f = model(data)
    t=recog(data,f)
    t = target_norm.back(t).cpu().numpy()
    # elapsed = record.time() - start_time
    # print(elapsed)

    xy, r, f = data.x[:, :3].cpu().numpy(), data.y.cpu().numpy(), f.cpu().numpy()
    #
    # # t = np.concatenate(valid_element[i])
    #
    xy = coords_norm.back(xy)
    r = fields_norm.back(r)
    f = fields_norm.back(f)
    plt.figure(1, figsize=(60, 20))
    plt.clf()
    visual.plot_fields(xy[0:1445, 1], xy[0:1445, 2],f[15*1445:16*1445,:], f[0*1445:1*1445,:])
    plt.savefig(path + "\\fieldpre_4-1.jpg")

    # fig, axs = plt.subplots(1, 12, figsize=(90, 10)) #3表示前3个性能参数 30是横坐标长度 10是纵坐标长度

    design_x = design_norm.back(design_x.cpu().numpy())

    # for task, ax in enumerate(axs):
    #     ax.plot(design_x[:, i], t[:, task], 'b')
    #     ax.set_title(f'Task {task + 1}')
    #     # Shade in confidence
    #     #ax.fill_between(design_x[:, i], lower[:, task], upper[:, task], alpha=0.5)
    #     #ax.legend(['Observed Data', 'Mean', 'Confidence'])
    # plt.savefig(path + '\\design-target.svg')
np.savetxt(path + '\\unsteady-design.csv', (design_x[:, 0], design_x[:, 1], design_x[:, 2], design_x[:, 6]),fmt='%.18e', delimiter=',')  # 输出非定常输入参数
np.savetxt(path + '\\unsteady-pre.csv', (t[:, 0],t[:, 1],t[:, 2],t[:, 3],t[:, 4],t[:, 8]), fmt='%.18e', delimiter=',') #输出性能参数预测值
#    np.savetxt(path + '\\location_' + str(i) + '-pre.csv', (design_x[:, i], t[:, 5], t[:, 6], t[:, 7], t[:, 9], t[:, 10], t[:, 11]), fmt='%.18e', delimiter=',')  # 输出位置预测值
