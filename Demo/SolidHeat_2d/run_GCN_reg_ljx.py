import os
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import time as record
from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
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
# shuffle_number = np.random.permutation(len(dataset)).tolist()
# dataset = dataset[shuffle_number]

# dataset = dataset[:1000]
n = int(len(dataset) * train_size)
n2 = int(len(dataset) * (1-valid_size))

train_dataset = dataset[:n]
valid_dataset = dataset[n2:]

# train_element = elements[:n]
# valid_element = elements[n:]

del dataset

train_loader = DataLoader(train_dataset, batch_size=64,  shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
visual_loader = DataLoader(valid_dataset[-50:], batch_size=1,  shuffle=False)

input_channels = 15 #12个设计变量+3个坐标
output_channels = 1 #输出温度场

planes = [input_channels,] + [32, 64, 128, 128, 128, 64, 32] + [output_channels]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model.SAGENet_U(planes).to(device)

recog = Model.Recognizer_SAGE_all([16, 6]).to(device)
# recog = Model.Recognizer_SAGE([4, 6]).to(device)

# recog = Model.Recognizer_SAGE([4, 6]).to(device) #输出功率、效率、最大温度、平均温度、最大应力及位置、最大应变及位置


epoch_start = 0
# model.load_state_dict(torch.load(os.path.join(path, 'latest_model.pth'))['model'])
# recog.load_state_dict(torch.load(os.path.join(path, 'latest_recog.pth'))['recog'])

# epoch_start = torch.load(os.path.join(path, 'latest_recog.pth'))['epoch']

optimizer_1 = torch.optim.Adamax(model.parameters(), lr=0.005)
optimizer_2 = torch.optim.Adamax(recog.parameters(), lr=0.002)

scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_1, milestones=[301, 401, 501], gamma=0.1)
scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_2, milestones=[301, 401, 501], gamma=0.1)
target_loss = nn.MSELoss().to(device)
fields_loss = Model.BoundaryMSE(device=device, weights=1.)

visual = Visual.matplotlib_vision(path)
#error = Model.prediction_error(device=device, nodes_num=336*5)

train_loss_log = []
valid_loss_log = []


for epoch in range(epoch_start, 501):

    start_time = record.time()

    target_log = []
    for it, data in enumerate(train_loader):
    # for it in range(401):

        # data = next(iter(train_loader))
        model.train()
        data = data.to(device)
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        f = model(data)

        if epoch < 200:
            t = recog(data, data.y)
        else:
            t = recog(data, f.detach())

        train_fields_loss = fields_loss(data.batch, f, data.y)
        train_fields_loss.backward()
        optimizer_1.step()

        train_target_loss = target_loss(t, data.t)
        train_target_loss.backward()
        optimizer_2.step()

        if (it % 50 == 0 and it > 0) : #训练集调小的时候此处iteration不到100
            # 验证
            data = next(iter(valid_loader)).to(device)
            model.eval()
            with torch.no_grad():
                data = data.to(device)
                f = model(data)
                valid_fields_loss = fields_loss(data.batch, f, data.y)
                t_ = recog(data, data.y)
                t__ = recog(data, f)
                valid_target_loss1 = target_loss(t_, data.t)
                valid_target_loss2 = target_loss(t__, data.t)

            t = target_norm.back(data.t).cpu()
            t_ = target_norm.back(t_).cpu()
            t__ = target_norm.back(t__).cpu()

            # _, t_ = error.target_error(t, t_)
            # t, t__ = error.target_error(t, t__)

            train_loss_log.append([train_fields_loss.item(), train_target_loss.item(),])
            valid_loss_log.append([valid_fields_loss.item(), valid_target_loss1.item(), valid_target_loss2.item()])
            target_log.append(np.concatenate((t.numpy(), t_.numpy(), t__.numpy()), axis=-1))

            elapsed = record.time() - start_time

            print(
                'epoch: %d / %d, iter: iter %d / %d,  Cost: %.2f, Lr: %.2e , '
                'train fields Loss = %.2e, test fields Loss = %.2e , '
                'train target Loss = %.2e, test target Loss = %.2e , '
                % (epoch, 501, it, len(train_loader), elapsed,
                   optimizer_1.state_dict()['param_groups'][0]['lr'],
                   train_loss_log[-1][0], valid_loss_log[-1][0],
                   train_loss_log[-1][1], valid_loss_log[-1][1],
                   )
            )

    # 学习率调整
    scheduler_1.step()
    scheduler_2.step()

    if epoch % 50 == 0:

        # 模型保存
        torch.save({'epoch': epoch, 'model': model.state_dict()}, os.path.join(path, 'latest_model.pth'))
        torch.save({'epoch': epoch, 'recog': recog.state_dict()}, os.path.join(path, 'latest_recog.pth'))

        plt.figure(10, figsize=(30, 10))
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.title('fields loss process')
        visual.plot_loss(np.arange(len(train_loss_log)), np.array(train_loss_log)[:, 0], label='train loss')
        visual.plot_loss(np.arange(len(train_loss_log)), np.array(valid_loss_log)[:, 0], label='valid loss')

        plt.subplot(1, 2, 2)
        plt.title('target loss process')
        visual.plot_loss(np.arange(len(train_loss_log)), np.array(train_loss_log)[:, 1], label='train loss')
        visual.plot_loss(np.arange(len(train_loss_log)), np.array(valid_loss_log)[:, 1], label='valid true loss')
        visual.plot_loss(np.arange(len(train_loss_log)), np.array(valid_loss_log)[:, 2], label='valid pred loss')

        plt.savefig(path + "\\loss.svg")

        np.savetxt(path + '\\train_loss.txt', np.array(train_loss_log))
        np.savetxt(path + '\\valid_loss.txt', np.array(valid_loss_log))


"""eval"""
# 损失评估
import tqdm
error_log_L1 = []
error_log_L2 = []
error = Model.prediction_error(device=device, nodes_num=0)
minibatch_iter = tqdm.tqdm(valid_loader, desc="Minibatch", leave=True)
for data in minibatch_iter:
    # data = next(iter(valid_loader)).to(device)
    data.to(device)
    model.eval()
    with torch.no_grad():
        # for data in loader:
        # data = loader[0]
        f = model(data)
        fields = data.y

        L1, L2, li = error.fields_error2(data.batch, fields, f)
    error_log_L2.append(L2)
    error_log_L1.append(L1)
error_log_L2 = np.concatenate(error_log_L2, axis=0)
error_log_L1 = np.concatenate(error_log_L1, axis=0)

np.savetxt(path + '\\L22.txt', np.array(error_log_L2))
np.savetxt(path + '\\L21.txt', np.array(error_log_L1))



train_preds_log = []
train_t_log = []
minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=True)
for data in minibatch_iter:
    # data = next(iter(train_loader)).to(device)
    data.to(device)
    model.eval()
    with torch.no_grad():
        train_f = model(data)
        train_preds = recog(data, train_f)
    train_preds = target_norm.back(train_preds).cpu()
    train_t = target_norm.back(data.t).cpu()
    train_preds_log.append(train_preds)
    train_t_log.append(train_t)
train_preds_log = np.concatenate(train_preds_log, axis=0)
train_t_log = np.concatenate(train_t_log, axis=0)
r2s_train = r2_score(train_preds_log, train_t_log, multioutput='raw_values')



valid_preds_log = []
valid_t_log = []
minibatch_iter = tqdm.tqdm(valid_loader, desc="Minibatch", leave=True)
for data in minibatch_iter:
    # data = next(iter(valid_loader)).to(device)
    data.to(device)
    model.eval()
    with torch.no_grad():
        valid_f = model(data)
        valid_preds = recog(data, valid_f)
    valid_preds = target_norm.back(valid_preds).cpu()
    valid_t = target_norm.back(data.t).cpu()

    valid_preds_log.append(valid_preds)
    valid_t_log.append(valid_t)

valid_preds_log = np.concatenate(valid_preds_log, axis=0)
valid_t_log = np.concatenate(valid_t_log, axis=0)
import math

r2s_valid = r2_score(valid_preds_log, valid_t_log, multioutput='raw_values')
mse_valid = mean_squared_error(valid_preds_log, valid_t_log, multioutput='raw_values')
mae_valid = mean_absolute_error(valid_preds_log, valid_t_log, multioutput='raw_values')

# np.savetxt(path + '\\MSE.txt', (mse_valid, mse_valid))
# np.savetxt(path + '\\MAE.txt', (mae_valid, mae_valid))
np.savetxt(path + '\\R2.txt', (r2s_train, r2s_valid))
np.savetxt(path + '\\valid.txt', np.concatenate((valid_t_log, valid_preds_log), axis = -1))


loss_log = []
target_log = []
for i, data in enumerate(train_loader):
        data.to(device)
        model.eval()
        with torch.no_grad():

            f = model(data)
            target_fields_loss = fields_loss(data.batch, f, data.y)
            t__ = recog(data, f)
            target_loss2 = target_loss(t__, data.t)

        loss_log.append([target_fields_loss.item(), target_loss2.item(),])

loss_log1 = np.mean(np.array(loss_log), axis=0)


loss_log = []
target_log = []
for i, data in enumerate(valid_loader):
        data.to(device)
        model.eval()
        with torch.no_grad():

            f = model(data)
            target_fields_loss = fields_loss(data.batch, f, data.y)
            t__ = recog(data, f)
            target_loss2 = target_loss(t__, data.t)

        loss_log.append([target_fields_loss.item(), target_loss2.item(),])

loss_log2 = np.mean(np.array(loss_log), axis=0)


np.savetxt(path + '\\train_valid_MSE.txt', (loss_log1, loss_log2))

##绘制预测性能误差

train_loss_log = []
valid_loss_log = []
target_log = []
import tqdm
minibatch_iter = tqdm.tqdm(valid_loader, desc="Minibatch", leave=True)

for data in minibatch_iter:
    # data = next(iter(train_loader)).to(device)
    data.to(device)
    model.eval()
    with torch.no_grad():
        time_1 = record.time()
        f = model(data)
        # valid_fields_loss = fields_loss(data.batch, f, data.y)
        t_ = recog(data, data.y)
        t__ = recog(data, f)
        time_2 = record.time()
        time_re = time_2 - time_1
        valid_target_loss1 = target_loss(t_, data.t)
        valid_target_loss2 = target_loss(t__, data.t)

    t = target_norm.back(data.t).cpu()
    t_ = target_norm.back(t_).cpu()
    t__ = target_norm.back(t__).cpu()
    error_p = (t__-t)/t
    target_log.append(np.concatenate((t.numpy(), t_.numpy(), t__.numpy(), error_p.numpy()), axis=-1))

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


