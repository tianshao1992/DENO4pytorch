import torch
import numpy as np
from torch_sparse import coalesce
import h5py
# from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


class PDataset(InMemoryDataset):
    def __init__(self, name, root, transform=None, pre_transform=None, pre_filter=None, if_download=False):
        self.name = name
        super(PDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, self.norms = read_data2(name)
        # self.data, self.slices, self.norms = read_data(self.name)

def read_data(name='temp_0.5span_limitss_ljx'):
    all = torch.load('data\\'+'GNN_' + name + '.pth', 'r')

    ind = all['ind']

    batch = all['batch']
    nodes = all['nodes'][:, :3]
    edges = all['edges'].transpose((1, 0))

    targets = all['target'][:len(ind), [0,1,2,3,4,8]]
        # attrs = all['nodes'][:, 2:]
    # # mdess 可以试下删除几何参数，几何参数在
    # mdess = all['design'][ind]

    attrs = all['nodes'][:, 3:]
    mdess = all['design'][:len(ind), :]
    c_min_max = np.stack([np.min(nodes, axis=0), np.max(nodes, axis=0)], axis=0)
    f_min_max = np.stack([np.min(attrs, axis=0), np.max(attrs, axis=0)], axis=0)
    d_min_max = np.stack([np.min(mdess, axis=0), np.max(mdess, axis=0)], axis=0)
    # c_norm 节点初始化
    c_norm = data_norm(c_min_max)
    f_norm = data_norm(f_min_max)
    d_norm = data_norm(d_min_max)
    t_norm = data_norm(targets)

    attrs = f_norm.norm(attrs)
    nodes = c_norm.norm(nodes)
    mdess = d_norm.norm(mdess)
    target = t_norm.norm(targets)

    # all = torch.load('data\\ANN_' + name + '.pth', 'r')
    # a = c_norm.norm(all['nodes'][39][:336, :2])
    # import matplotlib.pyplot as plt
    # plt.figure(1)
    # plt.scatter(a[:, 0], a[:, 1])
    # plt.figure(2)
    # a = c_norm.norm(a)
    # plt.scatter(a[:, 0], a[:, 1])
    # plt.figure(1)
    # a = c_norm.back(a)
    # plt.scatter(a[:, 0], a[:, 1])
    # plt.show()
    #
    batch = torch.tensor(batch, dtype=torch.long).squeeze()
    edges = torch.tensor(edges, dtype=torch.long)
    nodes = torch.tensor(nodes, dtype=torch.float32)
    # elems = torch.tensor(np.concatenate(elems, axis=0), dtype=torch.long)
    attrs = torch.tensor(attrs, dtype=torch.float32)

    mdess = torch.tensor(mdess[batch], dtype=torch.float32)

    nodes = torch.cat((nodes, mdess), dim=-1)

    target = torch.tensor(target, dtype=torch.float32)


    # import matplotlib.pyplot as plt
    # plt.figure(1, figsize=(20, 10))
    # plt.ion()
    # plt.clf()
    #
    # nodes = nodes.numpy()
    # attrs = attrs.numpy()
    # plt.scatter(nodes[:1000, 1], nodes[:1000, 0], c=attrs[:1000, -2], cmap="rainbow", marker='*', s=1.0, linewidths=2)
    # plt.axis('equal')
    # plt.colorbar()
    # plt.pause(0.1)
    #
    # plt.show()

    edge_attr = None

    num_nodes = nodes.shape[0]
    # edge_index, edge_attr = remove_self_loops(edges, edge_attr)
    edge_index, edge_attr = coalesce(edges, edge_attr, num_nodes, num_nodes)

    data = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr, y=attrs, pos=nodes[:, :3], t=target)
    # data.t = target

    data, slices = split(data, batch)
    torch.save((data, slices, c_min_max, f_min_max, d_min_max, targets), 'data\\'+'GNN_data_5000ljx_12-6.pth')


    return data, slices, (d_norm, c_norm, f_norm, t_norm)

def read_data2(data_name):
    all = torch.load('data\\GNN_data_'+data_name+'.pth')
    data = all[0]
    slices = all[1]
    c_min_max = all[2]
    f_min_max = all[3]
    d_min_max = all[4]
    targets = all[5]
    c_norm = data_norm(c_min_max)
    f_norm = data_norm(f_min_max)
    d_norm = data_norm(d_min_max)
    t_norm = data_norm(targets)
    return data, slices, (d_norm, c_norm, f_norm, t_norm)

class data_norm():

    def __init__(self, data, method="min-max", scale=1.0):
        axis = tuple(range(len(data.shape) - 1))
        self.method = method

        if method == "min-max":
            self.max = np.max(data, axis=axis)
            self.min = np.min(data, axis=axis)

        elif method == "mean-std":
            self.mean = np.mean(data, axis=axis)
            self.std = np.std(data, axis=axis)

        elif method == "scale":
            self.scale = scale


    def norm(self, x):
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = 2 * (x - torch.tensor(self.min, device=x.device)) \
                    / (torch.tensor(self.max, device=x.device) - torch.tensor(self.min, device=x.device) + 1e-10) - 1
            elif self.method == "mean-std":
                x = (x - torch.tensor(self.mean, device=x.device)) / (torch.tensor(self.std, device=x.device) + 1e-10)
            elif self.method == "scale":
                x = x / self.scale

        else:
            if self.method == "min-max":
                x = 2 * (x - self.min) / (self.max - self.min+1e-10) - 1
            elif self.method == "mean-std":
                x = (x - self.mean) / (self.std + 1e-10)
            elif self.method == "scale":
                x = x / self.scale

        return x

    def back(self, x):
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = (x + 1) / 2 * (torch.tensor(self.max, device=x.device)
                                   - torch.tensor(self.min, device=x.device) + 1e-10) + torch.tensor(self.min, device=x.device)
            elif self.method == "mean-std":
                x = x * (torch.tensor(self.std, device=x.device) + 1e-10) + torch.tensor(self.mean, device=x.device)
            elif self.method == "scale":
                x = x * self.scale
        else:
            if self.method == "min-max":
                x = (x + 1) / 2 * (self.max - self.min+1e-10) + self.min
            elif self.method == "mean-std":
                x = x * (self.std + 1e-10) + self.mean
            elif self.method == "scale":
                x = x * self.scale

        return x


def split(data, batch):
    node_slice = torch.cumsum(torch.bincount(batch), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.bincount(batch[row]), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = torch.bincount(batch)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.pos is not None:
        slices['pos'] = node_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    if data.t is not None:
        slices['t'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices


if __name__ == '__main__':
    read_data()
