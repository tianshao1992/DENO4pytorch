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
data_name = '10000_12-6'
path = 'results_ljx\\' + data_name + "_test_U" + str(train_size) + "depth8"
if not os.path.exists(path):
    os.makedirs(path)

input_channels = 15 #12个设计变量+3个坐标
output_channels = 1 #输出温度场

planes = [input_channels,] + [64, 128, 128, 128, 64] + [output_channels]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model.SAGENet(planes).to(device)
recog = Model.Recognizer_SAGE([4, 6]).to(device) #输出功率、效率、最大温度、平均温度、最大应力及位置、最大应变及位置

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

n2 = int(len(dataset) * (1-valid_size))
train_dataset = dataset[:n]
valid_dataset = dataset[n2:]
#
# train_element = elements[:n]
# valid_element = elements[n:]

#del dataset

train_loader = DataLoader(train_dataset, batch_size=16,  shuffle=True)
#valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=500, shuffle=False, drop_last=False)
visual_loader = DataLoader(valid_dataset[-50:], batch_size=1,  shuffle=False)
data = next(iter(valid_loader))

visual = Visual.matplotlib_vision(path)

with torch.no_grad():
    data = data.to(device)
    f = model(data)
    t_ = recog(data, data.y) #原始温度场
    t__ = recog(data, f)     #预测温度场

t = target_norm.back(data.t).cpu()
t_ = target_norm.back(t_).cpu()
t__ = target_norm.back(t__).cpu()

target_log = []
target_log.append(np.concatenate((t.numpy(), t_.numpy(), t__.numpy()), axis=-1))

loader = iter(visual_loader)

for i in range(1): #只画温度图
    model.eval()
    with torch.no_grad():
     # for data in loader:
        data = next(loader).to(device)
         # data = loader[0]
        f = model(data).detach()

        xy, r, f = data.x[:, :3].cpu().numpy(), data.y.cpu().numpy(), f.cpu().numpy()

         # t = np.concatenate(valid_element[i])

        xy = coords_norm.back(xy)
        r = fields_norm.back(r)
        f = fields_norm.back(f)

        plt.figure(1, figsize=(60, 20))
        plt.clf()
        visual.plot_fields(xy[:, 1], xy[:, 2], r, f)
        plt.savefig(path + "\\t_" + str(i) + ".jpg")

            # plt.figure(2, figsize=(30, 25))
            # plt.clf()
            # visual.plot_fields(xy[:, 0], xy[:, 1], r, f)
            # plt.savefig(path + "\\p_" + str(i) + "_d.jpg")

            #
            # plt.figure(3, figsize=(30, 30))
            # plt.clf()
            # visual.plot_fields_tr(xy[:, 0], xy[:, 1], r, f, xy[:336], triangles=t)
            # plt.savefig(path + "\\c_" + str(i) + ".jpg")
            #
            # plt.figure(4, figsize=(30, 25))
            # plt.clf()
            # visual.plot_fields_tr(xy[:, 0], xy[:, 1], r, f, xy[:336], triangles=t, xmin_max=True)
            # plt.savefig(path + "\\c_" + str(i) + "_d.jpg")


        target_log = np.concatenate(target_log, axis=0)

for i in (0,1,2,3,4,5):
#for i in (0, 1, 2, 3, 4,5,6,7,8,9,10,11):
    plt.figure(101, figsize=(30, 15))
    plt.clf()
    plt.subplot(231)
    visual.plot_regression(target_log[:, i], target_log[:, i + 6]) #12个性能参数（流场真实值预测）
    plt.subplot(232)
    visual.plot_regression(target_log[:, i], target_log[:, i + 12])  # 12个性能参数（流场预测值预测）
    if i == 0:
                plt.savefig(path + '\\effic_train.svg')
    if i == 1:
                plt.savefig(path + '\\power_train.svg')
    if i == 2:
                plt.savefig(path + '\\Tave_train.svg')
    if i == 3:
                plt.savefig(path + '\\Tmax_train.svg')
    if i == 4:
                plt.savefig(path + '\\stress_train.svg')
    # if i == 5:
    #             plt.savefig(path + '\\stress_x.svg')
    # if i == 6:
    #             plt.savefig(path + '\\stress_y.svg')
    # if i == 7:
    #             plt.savefig(path + '\\stress_z.svg')
    # if i == 9:
    #             plt.savefig(path + '\\strain_x.svg')
    # if i == 10:
    #             plt.savefig(path + '\\strain_y.svg')
    # if i == 11:
    #             plt.savefig(path + '\\strain_z.svg')
    else:
                plt.savefig(path + '\\strain_train.svg')
            # plt.show()
for i in (0, 1, 2, 3, 4, 5):
    r2 = r2_score(target_log[:, i], target_log[:, i+6])
    r2_pre = r2_score(target_log[:, i], target_log[:, i + 12])
    t_error = np.abs((target_log[:, i] - target_log[:, i+6]) / target_log[:, i])
    t_error_pre = np.abs((target_log[:, i] - target_log[:, i + 12]) / target_log[:, i])
    mean=np.mean(t_error)
    mean_pre=np.mean(t_error_pre)
    Lt_error = t_error.std(axis=0)
    Lt_error_pre = t_error_pre.std(axis=0)
    if i == 0:
        np.savetxt(path + '\\eff_error.csv', (mean,mean_pre,r2, r2_pre,Lt_error, Lt_error_pre), fmt='%.18e', delimiter=',')
    if i == 1:
        np.savetxt(path + '\\power_error.csv', (mean,mean_pre,r2, r2_pre,Lt_error, Lt_error_pre), fmt='%.18e', delimiter=',')
    if i == 2:
        np.savetxt(path + '\\Tave_error.csv', (mean,mean_pre,r2, r2_pre,Lt_error, Lt_error_pre), fmt='%.18e', delimiter=',')
    if i == 3:
        np.savetxt(path + '\\Tmax_error.csv', (mean,mean_pre,r2, r2_pre,Lt_error, Lt_error_pre), fmt='%.18e', delimiter=',')
    if i == 4:
        np.savetxt(path + '\\stress_error.csv', (mean,mean_pre,r2, r2_pre,Lt_error, Lt_error_pre), fmt='%.18e', delimiter=',')
    else:
        np.savetxt(path + '\\strain_error.csv', (mean,mean_pre,r2, r2_pre,Lt_error, Lt_error_pre), fmt='%.18e', delimiter=',')

np.savetxt(path + '\\targets_error.csv', (target_log), fmt='%.18e', delimiter=',')
