import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import SplineConv, GMMConv, GATConv, SAGEConv, PANConv, GCNConv
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GATNet(nn.Module):
    def __init__(self, planes):
        super(GATNet, self).__init__()
        self.planes = planes
        self.layers = nn.ModuleList()

        for i in range(len(self.planes) - 2):
            self.layers.append(GATConv(self.planes[i], self.planes[i + 1], heads=3, concat=False))
        self.layers.append(GATConv(self.planes[-2], self.planes[-1], heads=1))
        self.active = nn.GELU()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, edge_index)
            x = self.active(x)
        x = self.layers[-1](x, edge_index)
        return x

class GCNNet(nn.Module):
    def __init__(self, planes):
        super(GCNNet, self).__init__()
        self.planes = planes
        self.layers = nn.ModuleList()
        for i in range(len(self.planes) - 2):
            self.layers.append(GCNConv(self.planes[i], self.planes[i + 1]))
        self.layers.append(GCNConv(self.planes[-2], self.planes[-1]))
        self.active = nn.GELU()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, edge_index)
            x = self.active(x)
        x = self.layers[-1](x, edge_index)
        return x

class SAGENet(nn.Module):
    def __init__(self, planes):
        super(SAGENet, self).__init__()
        self.planes = planes
        self.layers = nn.ModuleList()
        for i in range(len(self.planes) - 2):
            self.layers.append(SAGEConv(self.planes[i], self.planes[i + 1]))
        self.layers.append(SAGEConv(self.planes[-2], self.planes[-1]))
        self.active = nn.GELU()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, edge_index)
            x = self.active(x)
        x = self.layers[-1](x, edge_index)
        return x

class SAGENet_U(nn.Module):
    def __init__(self, planes):
        super(SAGENet_U, self).__init__()
        self.planes = planes
        self.layers = nn.ModuleList()
        for i in range(len(self.planes) - 2):
            self.layers.append(SAGEConv(self.planes[i], self.planes[i + 1]))
        self.layers.append(SAGEConv(self.planes[-2], self.planes[-1]))
        self.half_layers = (len(self.layers)+1) // 2
        self.half_idx = (len(self.layers)+1) % 2
        self.all_layers = len(self.layers)
        self.active = nn.GELU()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h = []
        for i in range(self.half_layers-1):
            x = self.layers[i](x, edge_index)
            x = self.active(x)
            h.append(x)

        x = self.layers[self.half_layers - 1](x, edge_index)
        x = self.active(x)

        for i in range(self.half_layers, self.all_layers-1):
            x = self.layers[i](x, edge_index)
            x += h[self.all_layers-2-i]
            x = self.active(x)

        x = self.layers[-1](x, edge_index)

        return x



class PANnet(nn.Module):
    def __init__(self, planes):
        super(PANnet, self).__init__()
        self.planes = planes
        self.layers = nn.ModuleList()
        for i in range(len(self.planes) - 2):
            self.layers.append(PANConv(self.planes[i], self.planes[i + 1]))
        self.layers.append(PANConv(self.planes[-2], self.planes[-1]))
        self.active = nn.GELU()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, edge_index)
            x = self.active(x)
        x = self.layers[-1](x, edge_index)
        return x


class GMMNet(nn.Module):
    def __init__(self, planes):
        super(GMMNet, self).__init__()
        self.planes = planes
        self.layers = nn.ModuleList()
        for i in range(len(self.planes) - 2):
            self.layers.append(GMMConv(self.planes[i], self.planes[i + 1], dim=1, kernel_size=3))

        self.layers.append(GMMConv(self.planes[-2], self.planes[-1], dim=1, kernel_size=3))
        self.active = nn.GELU()

    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, edge_index, edge_attr)
            x = self.active(x)
        x = self.layers[-1](x, edge_index, edge_attr)
        return x



class GMMNet_U(nn.Module):
    def __init__(self, planes):
        super(GMMNet_U, self).__init__()
        self.planes = planes
        self.layers = nn.ModuleList()
        for i in range(len(self.planes) - 2):
            self.layers.append(GMMConv(self.planes[i], self.planes[i + 1], dim=1, kernel_size=3))
        self.layers.append(GMMConv(self.planes[-2], self.planes[-1], dim=1, kernel_size=3))
        self.half_layers = (len(self.layers)+1) // 2
        self.half_idx = (len(self.layers)+1) % 2
        self.all_layers = len(self.layers)
        self.active = nn.GELU()

    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = []
        for i in range(self.half_layers-1):
            x = self.layers[i](x, edge_index, edge_attr)
            x = self.active(x)
            h.append(x)

        x = self.layers[self.half_layers - 1](x, edge_index, edge_attr)
        x = self.active(x)

        for i in range(self.half_layers, self.all_layers-1):
            x = self.layers[i](x, edge_index, edge_attr)
            x += h[self.all_layers-2-i]
            x = self.active(x)

        x = self.layers[-1](x, edge_index, edge_attr)
        return x



class SPLNet(nn.Module):
    def __init__(self, planes):
        super(SPLNet, self).__init__()
        self.planes = planes
        self.layers = nn.ModuleList()
        for i in range(len(self.planes) - 2):
            self.layers.append(SplineConv(self.planes[i], self.planes[i + 1], dim=1, kernel_size=3))
        self.layers.append(SplineConv(self.planes[-2], self.planes[-1], dim=1, kernel_size=3))
        self.active = nn.GELU()


    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i in range(len(self.layers)-1):
            x = self.layers[i](x, edge_index, edge_attr)
            x = self.active(x)
        x = self.layers[-1](x, edge_index, edge_attr)
        return x


class SPLNet_U(nn.Module):
    def __init__(self, planes):
        super(SPLNet_U, self).__init__()
        self.planes = planes
        self.layers = nn.ModuleList()
        for i in range(len(self.planes) - 2):
            self.layers.append(SplineConv(self.planes[i], self.planes[i + 1], dim=1, kernel_size=3))
        self.layers.append(SplineConv(self.planes[-2], self.planes[-1], dim=1, kernel_size=3))
        self.half_layers = (len(self.layers)+1) // 2
        self.half_idx = (len(self.layers)+1) % 2
        self.all_layers = len(self.layers)
        self.acitve = nn.ReLU(inplace=True)
    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = []
        for i in range(self.half_layers-1):
            x = self.layers[i](x, edge_index, edge_attr)
            x = self.active(x)
            h.append(x)
        x = self.layers[self.half_layers - 1](x, edge_index, edge_attr)
        x = self.acitve(x)
        for i in range(self.half_layers, self.all_layers-1):
            x = self.layers[i](x, edge_index, edge_attr)
            x += h[self.all_layers-2-i]
            x = self.acitve(x)
        x = self.layers[-1](x, edge_index, edge_attr)
        return x

class BoundaryMSE(nn.Module):

    def __init__(self, device, weights_num=336*2, weights=1):
        super(BoundaryMSE, self).__init__()

        self.weights_num = weights_num
        self.device = device
        self.weights = weights

    def forward(self, batch, pred, true):

        weight = torch.ones_like(true)

        num_case = batch.max() + 1
        boundary_index = torch.arange(self.weights_num, device=self.device) \
                      * torch.ones((num_case, 1), dtype=torch.int32, device=self.device)
        boundary_index[1:, :] += torch.cumsum(torch.bincount(batch), dim=0)[:-1].unsqueeze(1)
        boundary_index = boundary_index.reshape(-1)
        weight[boundary_index] = self.weights

        loss = torch.mean(weight*(pred - true)**2)

        return loss


class prediction_error(object):

    def __init__(self, nodes_num, device):

        self.device = device
        self.nodes_num = nodes_num

    def fields_error(self, batch, true, pred):
        ind = torch.cumsum(torch.bincount(batch), dim=0)
        ind = [0, ] + ind.cpu().numpy().tolist()

        L1_error, L2_error, Li_error = [], [], []

        for i in range(len(ind) - 1):

            err = pred[ind[i]:ind[i + 1]] - true[ind[i]:ind[i + 1]]
            L1_error.append(torch.mean(torch.abs(err), dim=0))
            L2_error.append(torch.mean(torch.square(err), dim=0))
            Li_error.append(torch.max(torch.abs(err), dim=0)[0])


        return torch.stack(L1_error, dim=0).cpu().numpy(), torch.stack(L2_error, dim=0).cpu().numpy(),\
               torch.stack(Li_error, dim=0).cpu().numpy()


    def fields_error2(self, batch, true, pred):
        ind = torch.cumsum(torch.bincount(batch), dim=0)
        ind = [0, ] + ind.cpu().numpy().tolist()

        L1_error, L2_error, Li_error = [], [], []

        for i in range(len(ind) - 1):

            err = (pred[ind[i]:ind[i + 1]] - true[ind[i]:ind[i + 1]])
            L1_error.append(torch.mean(torch.abs(err), dim=0))
            L2_error.append(torch.mean(torch.square(err), dim=0))
            Li_error.append(torch.max(torch.abs(err), dim=0)[0])


        return torch.stack(L1_error, dim=0).cpu().numpy(), torch.stack(L2_error, dim=0).cpu().numpy(),\
               torch.stack(Li_error, dim=0).cpu().numpy()

    def bounds_error(self, batch, true, pred):
        num_case = batch.max() + 1
        bounds_index = torch.arange(self.nodes_num, device=self.device) \
                      * torch.ones((num_case, 1), dtype=torch.int32, device=self.device)
        bounds_index[1:, :] += torch.cumsum(torch.bincount(batch), dim=0)[:-1].unsqueeze(1)
        bounds_index = bounds_index.reshape(-1)

        true = true[bounds_index].reshape((num_case, self.nodes_num, -1))
        pred = pred[bounds_index].reshape((num_case, self.nodes_num, -1))
        err = pred - true
        L1_error = torch.mean(err.abs(), dim=1).cpu().numpy()
        L2_error = torch.mean(err**2, dim=1).cpu().numpy()
        Li_error = torch.max(err.abs(), dim=1)[0].cpu().numpy()

        return L1_error, L2_error, Li_error


    def design_para(self, batch, true):
        ind = torch.cumsum(torch.bincount(batch), dim=0)
        ind = [0, ] + ind.cpu().numpy().tolist()

        design = []

        for i in range(len(ind) - 1):

            design_para = true[ind[i]:ind[i + 1]][0, 3:]

            design.append(design_para)

        return torch.stack(design, dim=0).cpu().numpy()


    def target_error(self, true, pred):
        err = (pred - true) / true
        err_index = torch.abs(err[:, 0]) > 0.010
        pred[err_index, 0] = true[err_index, 0] + err[err_index, 0] * true[err_index, 0] / 3.

        # err = (pred - true) / true

        return true.cpu(), pred.cpu()

class Recognizer(nn.Module):
    def __init__(self, planes):
        super(Recognizer, self).__init__()
        self.conv1 = SAGEConv(planes[0], 64)
        self.pool1 = TopKPooling(64, ratio=0.8)
        self.conv2 = SAGEConv(64, 64)
        self.pool2 = TopKPooling(64, ratio=0.8)
        self.conv3 = SAGEConv(64, 64)
        self.pool3 = TopKPooling(64, ratio=0.8)
        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, 64)
        self.lin3 = torch.nn.Linear(64, planes[-1])
        self.active = nn.GELU()

    def forward(self, data, fields):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = torch.cat((x[:, :3], fields), dim=-1)
        x = self.active(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.active(self.conv2(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.active(self.conv3(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2 + x3

        x = self.active(self.lin1(x))
        x = self.active(self.lin2(x))
        x = self.lin3(x)

        return x

class Recognizer_GMM(nn.Module):
    def __init__(self, planes):
        super(Recognizer_GMM, self).__init__()

        self.conv1 = GMMConv(planes[0], 64, dim=1, kernel_size=3)
        self.pool1 = TopKPooling(64, ratio=0.8)
        self.conv2 = GMMConv(64, 64, dim=1, kernel_size=3)
        self.pool2 = TopKPooling(64, ratio=0.8)
        self.conv3 = GMMConv(64, 64, dim=1, kernel_size=3)
        self.pool3 = TopKPooling(64, ratio=0.8)
        self.conv4 = GMMConv(64, 64, dim=1, kernel_size=3)
        self.pool4 = TopKPooling(64, ratio=0.8)
        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, 64)
        self.lin3 = torch.nn.Linear(64, planes[-1])
        self.active = nn.ReLU(inplace=True)

    def forward(self, data, fields):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        x = torch.cat((x[:, :3], fields), dim=-1)

        x = self.active(self.conv1(x, edge_index, edge_attr))

        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.active(self.conv2(x, edge_index, edge_attr))

        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.active(self.conv3(x, edge_index, edge_attr))

        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.active(self.conv4(x, edge_index, edge_attr))

        x, edge_index, edge_attr, batch, _, _ = self.pool4(x, edge_index, edge_attr, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2 + x3 + x4

        x = self.active(self.lin1(x))
        x = self.active(self.lin2(x))
        x = self.lin3(x)

        return x


class Recognizer_SPL(nn.Module):
    def __init__(self, planes):
        super(Recognizer_SPL, self).__init__()
        self.conv1 = SplineConv(planes[0], 64, dim=1, kernel_size=3)
        self.pool1 = TopKPooling(64, ratio=0.8)
        self.conv2 = SplineConv(64, 64, dim=1, kernel_size=3)
        self.pool2 = TopKPooling(64, ratio=0.8)
        self.conv3 = SplineConv(64, 64, dim=1, kernel_size=3)
        self.pool3 = TopKPooling(64, ratio=0.8)
        self.conv4 = SplineConv(64, 64, dim=1, kernel_size=3)
        self.pool4 = TopKPooling(64, ratio=0.8)
        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, 64)
        self.lin3 = torch.nn.Linear(64, planes[-1])
        self.active = nn.GELU()
    def forward(self, data, fields):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = torch.cat((x[:, :3], fields), dim=-1)
        x = self.active(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.active(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.active(self.conv3(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.active(self.conv4(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool4(x, edge_index, edge_attr, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2 + x3 + x4

        x = self.active(self.lin1(x))
        x = self.active(self.lin2(x))
        x = self.lin3(x)

        return x


class Recognizer_SAGE(nn.Module):
    def __init__(self, planes):
        super(Recognizer_SAGE, self).__init__()
        self.conv1 = SAGEConv(planes[0], 64)
        self.pool1 = TopKPooling(64, ratio=0.8)
        self.conv2 = SAGEConv(64, 64)
        self.pool2 = TopKPooling(64, ratio=0.8)
        self.conv3 = SAGEConv(64, 64)
        self.pool3 = TopKPooling(64, ratio=0.8)
        self.conv4 = SAGEConv(64, 64)
        self.pool4 = TopKPooling(64, ratio=0.8)
        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, 64)
        self.lin3 = torch.nn.Linear(64, planes[-1])
        self.active = nn.GELU()
    def forward(self, data, fields):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = torch.cat((x[:,:3], fields), dim=-1)
        x = self.active(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.active(self.conv2(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.active(self.conv3(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.active(self.conv4(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool4(x, edge_index, edge_attr, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2 + x3 + x4

        x = self.active(self.lin1(x))
        x = self.active(self.lin2(x))
        x = self.lin3(x)

        return x

class Recognizer_SAGE_all(nn.Module):
    def __init__(self, planes):
        super(Recognizer_SAGE_all, self).__init__()
        self.conv1 = SAGEConv(planes[0], 64)
        self.pool1 = TopKPooling(64, ratio=0.8)
        self.conv2 = SAGEConv(64, 64)
        self.pool2 = TopKPooling(64, ratio=0.8)
        self.conv3 = SAGEConv(64, 64)
        self.pool3 = TopKPooling(64, ratio=0.8)
        self.conv4 = SAGEConv(64, 64)
        self.pool4 = TopKPooling(64, ratio=0.8)
        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, 64)
        self.lin3 = torch.nn.Linear(64, planes[-1])
        self.active = nn.GELU()
    def forward(self, data, fields):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = torch.cat((x[:,:], fields), dim=-1) ##这里把所有设计变量都加上了
        x = self.active(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.active(self.conv2(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.active(self.conv3(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.active(self.conv4(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool4(x, edge_index, edge_attr, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2 + x3 + x4

        x = self.active(self.lin1(x))
        x = self.active(self.lin2(x))
        x = self.lin3(x)

        return x


class Recognizer_GAT(nn.Module):
    def __init__(self, planes):
        super(Recognizer_GAT, self).__init__()
        self.conv1 = GATConv(planes[0], 64, heads=3, concat=False)
        self.pool1 = TopKPooling(64, ratio=0.8)
        self.conv2 = GATConv(64, 64, heads=3, concat=False)
        self.pool2 = TopKPooling(64, ratio=0.8)
        self.conv3 = GATConv(64, 64, heads=3, concat=False)
        self.pool3 = TopKPooling(64, ratio=0.8)
        self.conv4 = GATConv(64, 64, heads=3, concat=False)
        self.pool4 = TopKPooling(64, ratio=0.8)
        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, 64)
        self.lin3 = torch.nn.Linear(64, planes[-1])
        self.active = nn.GELU()
    def forward(self, data, fields):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = torch.cat((x[:, :3], fields), dim=-1)
        x = self.active(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.active(self.conv2(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.active(self.conv3(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.active(self.conv4(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool4(x, edge_index, edge_attr, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2 + x3 + x4

        x = self.active(self.lin1(x))
        x = self.active(self.lin2(x))
        x = self.lin3(x)

        return x


class Dynamicor(nn.Module):

    def __init__(self, device, nodes_num=336):
        super(Dynamicor, self).__init__()

        self.device = device
        self.nodes_num = nodes_num
        self.ind1 = [i for i in range(nodes_num)]
        self.ind2 = [i for i in range(1, nodes_num)]
        self.ind2.append(0)

        self.acoustic = np.sqrt(1.4*287*300)
        self.density = 1.225
        self.miu = 1.9e-5
        self.alpha = 6 * np.pi / 180
        self.rotat = torch.tensor(np.array([[0, -1], [1, 0]]), dtype=torch.float, device=self.device)


    def forward(self, batch, coords, fields, design):

        num_case = batch.max() + 1
        foils_index = torch.arange(self.nodes_num, device=self.device) \
                      * torch.ones((num_case, 1), dtype=torch.int32, device=self.device)
        foils_index[1:, :] += torch.cumsum(torch.bincount(batch), dim=0)[:-1].unsqueeze(1)
        foils_index = foils_index.reshape(-1)
        design = design[foils_index].reshape((num_case, self.nodes_num, -1))

        foils_nodes0 = coords[foils_index].reshape((num_case, self.nodes_num, -1))
        foils_field0 = fields[foils_index].reshape((num_case, self.nodes_num, -1))

        foils_index = foils_index + self.nodes_num
        foils_nodes1 = coords[foils_index].reshape((num_case, self.nodes_num, -1))
        foils_field1 = fields[foils_index].reshape((num_case, self.nodes_num, -1))

        pt = foils_field0[:, :, 0]
        pt_ave = (pt[:, self.ind1] + pt[:, self.ind1]) / 2.

        # edge_middle = 0.5 * (foils_nodes0[:, self.ind1] + foils_nodes0[:, self.ind2])
        T_vector = foils_nodes0[:, self.ind2] - foils_nodes0[:, self.ind1]
        N_vector = torch.matmul(T_vector, self.rotat)
        T_norm = torch.norm(T_vector, dim=-1)
        N_norm = torch.norm(N_vector, dim=-1)

        Pt_n = pt_ave * T_norm
        Px = Pt_n * N_vector[:, :, 0] / N_norm
        Py = Pt_n * N_vector[:, :, 1] / N_norm
        # Mz = - Fx * (edge_middle[:, :, 1] - center[:, :, 1]) + Fy * (edge_middle[:, :, 0] - center[:, :, 0])

        ut = foils_field1[:, :, 2:]
        du = (ut[:, :, 0] * T_vector[:, :, 0] + ut[:, :, 1] * T_vector[:, :, 1]) / T_norm
        delta = torch.norm(foils_nodes1 - foils_nodes0, dim=-1)
        tau = self.miu * du / delta
        tau_ave = (tau[:, self.ind1] + tau[:, self.ind2]) * 0.5
        T_n = tau_ave * T_norm
        Tx = 50*T_n * T_vector[:, :, 0] / T_norm
        Ty = 50*T_n * T_vector[:, :, 1] / T_norm

        Fx = torch.sum(-Px, dim=1) + torch.sum(Tx, dim=1)
        Fy = torch.sum(-Py, dim=1) + torch.sum(Ty, dim=1)
        # Mz = torch.sum(Mz, dim=1)

        Ma, af = torch.mean(design[:, :, 3], dim=1)*0.3+0.3, torch.mean(design[:, :, 4], dim=1)*self.alpha
        Fx = Fx * torch.cos(af) + Fy * torch.sin(af)
        Fy = Fy * torch.cos(af) - Fx * torch.sin(af)

        velocity = self.acoustic * Ma
        Cd = Fx / (0.5*self.density*velocity**2)
        Cl = Fy / (0.5*self.density*velocity**2)

        return torch.stack((Cd, Cl), dim=-1)

if __name__ == '__main__':

    import GCN_data as Rdataset
    from torch_geometric.data import DataLoader
    import torch_geometric.transforms as T

    transform = T.Distance(cat=False)
    dataset = Rdataset.PDataset('Airfoils_unstruct', root='', transform=transform, if_download=False)
    design_norm = dataset.norms[0]
    coords_norm = dataset.norms[1]
    fields_norm = dataset.norms[2]
    import torch
    all = torch.load('data\\GCN_all.pth', 'r')
    elements = all['elems']
    del all

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dynamic = Dynamicor(336, device=device)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    data = next(iter(loader))

    coords = data.x[:, :3]
    fields = data.y
    batch = data.batch
    design = data.x[:, 2:]

    coords = coords_norm.back(coords)
    fields = fields_norm.back(fields)
    design = design_norm.back(design)

    c = dynamic(batch, coords, fields, design)