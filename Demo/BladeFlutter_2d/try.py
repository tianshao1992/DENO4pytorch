import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F




data_file = os.path.join('data', "flutter_bld_res1_1-300.mat")

datamat = h5py.File(data_file)
bld_fields = [datamat[element[0]][:] for element in datamat['NEW_bld_fields']]
fields = [torch.tensor(elem[:, 1:4], dtype=torch.float32) * 1000 for elem in bld_fields] # m 转为mm
coords = [torch.tensor(elem[:, 4:], dtype=torch.float32) for elem in bld_fields]

field = torch.tensor(bld_fields[0][None,...], dtype=torch.float32)
uniform_field = F.interpolate(field, [164, 36])

field = field.numpy()
uniform_field = uniform_field.numpy()

plt.ion()
figure, axs = plt.subplots(2, 1, figsize=(10, 20), dpi=300, num=100, subplot_kw={"projection": "3d"})
axs[0].scatter3D(field[0, 1, :, :], field[0, 2, :, :],field[0, 3, :, :], c=field[0, 4, :, :],)
axs[1].scatter3D(uniform_field[0, 1, :, :], uniform_field[0, 2, :, :],uniform_field[0, 3, :, :], c=uniform_field[0, 4, :, :],)
plt.ioff()