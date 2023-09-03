# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 03:33:23 2021

@author: admin
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io
import os

torch.manual_seed(123)
np.random.seed(123)

# os.chdir(r'C:\BaiduNetdiskDownload\G256to32_LES_newdata_50group')
################################################################
# 4d fourier layers

class SpectralConv4d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, modes4):
        super(SpectralConv4d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = min(modes4, 3//2+1)

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul4d(self, input, weights):
        # (batch, in_channel, x,y,z,t ), (in_channel, out_channel, x,y,z,t) -> (batch, out_channel, x,y,z,t)
        return torch.einsum("bixyzt,ioxyzt->boxyzt", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-4,-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-4), x.size(-3), x.size(-2), x.size(-1), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4] = \
            self.compl_mul4d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4], self.weights1)
        
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4] = \
            self.compl_mul4d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4] = \
            self.compl_mul4d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4] = \
            self.compl_mul4d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-4), x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO4d(nn.Module):
    def __init__(self, modes1, modes2, modes3, modes4, width):
        super(FNO4d, self).__init__()

        """
        input: the solution of the first 5 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, z=64, dim=3, c=5+3)
        output: the solution of the next  timestep
        output shape: (batchsize, x=64, y=64, z=64, dim=3, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.width = width
        self.fc0 = nn.Linear(9, self.width)
        # input channel is 5: the solution of the first 5 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.conv1 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.conv2 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.conv3 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        # self.conv4 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        # self.conv5 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)	
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        # self.w4 = nn.Conv1d(self.width, self.width, 1)
        # self.w5 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z, size_w = x.shape[1], x.shape[2], x.shape[3], x.shape[4]

        grid = self.get_grid(batchsize, size_x, size_y, size_z, size_w, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 5, 1, 2, 3, 4)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_w)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_w)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_w)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_w)
        x = x1 + x2
        # x = F.relu(x)
        
        # x1 = self.conv4(x)
        # x2 = self.w4(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_w)
        # x = x1 + x2
        # x = F.relu(x)
        
        # x1 = self.conv5(x)
        # x2 = self.w5(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_w)
        # x = x1 + x2
        
        

        x = x.permute(0, 2, 3, 4, 5, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, batchsize, size_x, size_y, size_z, size_w, device ):
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, size_w, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1, 1).repeat([batchsize, size_x, 1, size_z, size_w, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1, 1).repeat([batchsize, size_x, size_y, 1, size_w, 1])
        gridw = torch.tensor(np.linspace(0, 1, size_w), dtype=torch.float)
        gridw = gridw.reshape(1, 1, 1, 1, size_w, 1).repeat([batchsize, size_x, size_y, size_z, 1, 1])

        return torch.cat((gridx, gridy, gridz, gridw), dim=-1).to(device)

# input size should be [bs,64,64,64,3,5]
################################################################
# configs
################################################################
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
#-------------------------------------------------------------------------------需要调节的参数
#tunning3
modes = 16
width = 32
epochs = 100
learning_rate = 0.001
weight_decay_value = 1e-11
#网络层数
#显卡内存有没有爆掉cd C:\Program Files\NVIDIA Corporation\NVSMI，nvidia-smi
#保存模型的文件名
#---------------------------------------------------------------------------------------------

batch_size = 2
scheduler_step = 30
scheduler_gamma = 0.5  #每隔100步学习率衰减一半

print(epochs, learning_rate, scheduler_step, scheduler_gamma)


runtime = np.zeros(2, )
t1 = default_timer()


################################################################
# load data
################################################################
# input size should be [bs,64,64,64,3,5]
# output size should be [bs,64,64,64,3,1]

#------------------------------------------------------------这4行把(3000，64，64，64，3)的数据转成32的
# vor_data = np.load('./3d_vor_3000step_gap500_64.npy') #
# vor_data = torch.from_numpy(vor_data) #转成torch格式[3000, 64, 64, 64, 3]
#换位置才能对后三做池化，再换回来
# vor_data_downsample = nn.AvgPool3d(2, stride=2)(vor_data.permute(0,4,1,2,3)).permute(0,2,3,4,1)
# vor_data = vor_data_downsample
#-------------------------------------------------上面4行运行一次后保存出数据，下次直接加载数据
vor_data = np.load('data/HIT_vel_50g_600p_gap200_32.npy') #
vor_data = vor_data[0:45,...]
# vor_data2 = np.load('./FGR1_nofilter_vel_50g_600p_gap200_LES_32_add10more.npy') #
# vor_data2 = vor_data2[0:5,:,:,:,:,:] #用前45组训练，后5组测试
# vor_data3 = vor_data[0:20:2,:,:,:,:,:] 

vor_data = torch.from_numpy(vor_data) #
# vor_data2 = torch.from_numpy(vor_data2) #
# vor_data3 = torch.from_numpy(vor_data3) #

# vor_data = torch.cat([vor_data,vor_data2,vor_data3],dim=0)



# vor_data = vor_data[0:50,...]# 只取前50个样本来用，以防数据太大

# np.save('3d_turbulence_data_32',np.array(vor_data_downsample))
# given first 5 steps (T_in), predict the prospective 5 steps (T_out)
#prepare training and test data
# input previous 5 steps, output next step

input_list = []
output_list = []
# for j in range(vor_data.shape[0]):
#     for i in range(595):
#         input_list.append(vor_data[j,i:i+5,...])
#         output_list.append(vor_data[j,i+5,...])
# for j in range(vor_data.shape[0]):
#     for i in range(int(vor_data.shape[1]/6)):
#         # print(i)
#         input_list.append(vor_data[j,6*i:6*i+5,...])
#         output_list.append(vor_data[j,6*i+5,...])

for j in range(vor_data.shape[0]):
    for i in range(595):
        # print(i)
        input_list.append(vor_data[j,i:i+5,...])
        output_6m5 = (vor_data[j,i+5,...]-vor_data[j,i+4,...])
        output_list.append(output_6m5) 
        # output_list.append(vor_data[j,6*i+5,...])                
### switch dimension
# input size should be [bs,64,64,64,3,5]
# output size should be [bs,64,64,64,3]        
input_set = torch.stack(input_list) # torch.Size([2250, 5, 64, 64, 64, 3])
output_set = torch.stack(output_list) # torch.Size([2250, 32, 32, 32, 3])
input_set = input_set.permute(0,2,3,4,5,1) #torch.Size([2250, 64, 64, 64, 3, 5])


full_set = torch.utils.data.TensorDataset(input_set, output_set)
train_dataset, test_dataset = torch.utils.data.random_split(full_set, [int(0.8*len(full_set)), 
                                                                       len(full_set)-int(0.8*len(full_set))])

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

################################################################
# training and evaluation
################################################################
model = FNO4d(modes, modes, modes, modes, width).to(device)

#model = nn.DataParallel(model)
#model.to(device)
# model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

print(count_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

mse_train = []
mse_test = []


myloss = LpLoss()
# myloss = torch.nn.MSELoss(reduction='mean')
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    for xx, yy in train_loader:
        
        xx = xx.to(device)
        yy = yy.to(device)
        im = model(xx).to(device)
        
        train_loss = myloss(im.reshape(im.shape[0], -1), yy.reshape(yy.shape[0], -1))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    mse_train.append(train_loss.item())
        

    with torch.no_grad():
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)

            im = model(xx).to(device)
            test_loss = myloss(im.reshape(im.shape[0], -1), yy.reshape(yy.shape[0], -1))
        mse_test.append(test_loss.item())

    t2 = default_timer()
    
    
    print(ep, "%.2f" % (t2 - t1), 'train_loss: {:.4f}'.format(train_loss.item()), 
          'test_loss: {:.4f}'.format(test_loss.item()))

MSE_save=np.dstack((mse_train,mse_test)).squeeze()
np.savetxt('./20360.001_1e-11_4layer.dat',MSE_save,fmt="%16.7f")
# redefine retive error function


torch.save(model.state_dict(), '20360.001_1e-11_4layer.pth') #注意修改保存模型格式

# model = FNO4d(modes, modes, modes, modes, width).to(device)
# PATH = './trained_FNO4D_gap100.pth'
# model.load_state_dict(torch.load(PATH))
# model.eval()