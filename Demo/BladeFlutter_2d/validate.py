import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from gnn.GraphNets import KernelNN3, GMMNet
from Utilizes.loss_metrics import FieldsLpLoss
from Utilizes.visual_data import MatplotlibVision
from Utilizes.process_data import DataNormer
import matplotlib.pyplot as plt
import time
import os
import h5py


def inference(dataloader, netmodel, device):
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """
    if 'FNO' in
    with torch.no_grad():
        data = next(iter(dataloader))
        data = data.to(device)
        pred = netmodel(data)

    return data.x.cpu().numpy(), data.y.cpu().numpy(), pred.cpu().numpy()





if __name__ == "__main__":
    work_path = os.path.join('work', 'UNetForPTE', 'valid')
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    ntrain = 200
    batch_size = 32
    data_file = os.path.join('data', "flutter_bld_res1_1-300.mat")

    datamat = h5py.File(data_file)
    bld_fields= []
    index = []
    for ind, element in enumerate(datamat['NEW_bld_fields']):
        if np.size(datamat[element[0]][:]) > 10:
            bld_fields.append(datamat[element[0]][:])
            index.append(ind)

    all_fields = torch.cat([F.interpolate(torch.tensor(ff[None, ...],dtype=torch.float32), [164, 36]) for ff in bld_fields], dim=0)
    fields = torch.permute(all_fields[:, -6:-3, ...], (0, 2, 3, 1))
    coords = torch.permute(all_fields[:, 1:4, ...], (0, 2, 3, 1))
    # bld_elems = [datamat[element[0]][:] for element in datamat['NEW_bld_nodes']] # 在图卷积中使用

    design1 = torch.tensor(np.transpose(datamat['boundaries'], (1, 0)).squeeze()[:, 3:5], dtype=torch.float32)
    design2 = torch.tensor(np.transpose(datamat['geometries'], (1, 0)).squeeze()[:, 2:4], dtype=torch.float32)
    design = torch.cat([design1, design2], 1)
    design = torch.tile(design[index, None, None, :], (1, coords[0].shape[0], coords[0].shape[1], 1))
    input = torch.concat([coords, design], dim=-1)
    output = fields

    train_x = input[:ntrain, ...]
    train_y = output[:ntrain, ...]
    valid_x = input[ntrain:, ...]
    valid_y = output[ntrain:, ...]

    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    valid_x = x_normalizer.norm(valid_x)

    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    valid_y = y_normalizer.norm(valid_y)

    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                               batch_size=batch_size, shuffle=False, drop_last=True)



