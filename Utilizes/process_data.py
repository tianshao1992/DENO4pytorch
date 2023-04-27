#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2022/11/27 0:27
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : process_data.py
"""
import os.path
import numpy as np
import scipy
import scipy.io as sio
import torch
import h5py

import sklearn.metrics
from scipy.ndimage import gaussian_filter
from torch_geometric.data import Data



class DataNormer():
    """
        data normalization at last dimension
    """

    def __init__(self, data, method="min-max", axis=None):
        if axis is None:
            axis = tuple(range(len(data.shape) - 1))
        self.method = method
        if method == "min-max":
            self.max = np.max(data, axis=axis)
            self.min = np.min(data, axis=axis)

        elif method == "mean-std":
            self.mean = np.mean(data, axis=axis)
            self.std = np.std(data, axis=axis)

    def norm(self, x):
        """
            input tensors
        """
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = 2 * (x - torch.tensor(self.min, device=x.device)) \
                    / (torch.tensor(self.max, device=x.device) - torch.tensor(self.min, device=x.device) + 1e-10) - 1
            elif self.method == "mean-std":
                x = (x - torch.tensor(self.mean, device=x.device)) / (torch.tensor(self.std + 1e-10, device=x.device))
        else:
            if self.method == "min-max":
                x = 2 * (x - self.min) / (self.max - self.min + 1e-10) - 1
            elif self.method == "mean-std":
                x = (x - self.mean) / (self.std + 1e-10)

        return x

    def back(self, x):
        """
            input tensors
        """
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = (x + 1) / 2 * (torch.tensor(self.max, device=x.device)
                                   - torch.tensor(self.min, device=x.device) + 1e-10) + torch.tensor(self.min,
                                                                                                     device=x.device)
            elif self.method == "mean-std":
                x = x * (torch.tensor(self.std + 1e-10, device=x.device)) + torch.tensor(self.mean, device=x.device)
        else:
            if self.method == "min-max":
                x = (x + 1) / 2 * (self.max - self.min + 1e-10) + self.min
            elif self.method == "mean-std":
                x = x * (self.std + 1e-10) + self.mean
        return x
    def save(self,save_path):
        import pickle
        with open(os.path.join(save_path,'norm.pkl'), 'wb') as f:
            pickle.dump(self, f)

    def load(self,save_path):
        import pickle
        with open(os.path.join(save_path,'norm.pkl'), 'rb') as f:
            self = pickle.load(f)


# reading data
class MatLoader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatLoader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):

        try:
            self.data = sio.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class SquareMeshGenerator(object):
    # 在多维空间中获得正交网格
    def __init__(self, real_space, mesh_size):
        super(SquareMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.s = mesh_size[0]

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        rbf = np.exp(-pwd ** 2 / sigma ** 2)
        sample = np.random.binomial(1, rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def get_grid(self):
        return torch.tensor(self.grid, dtype=torch.float)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            else:
                edge_attr = np.zeros((self.n_edges, 3 * self.d))
                edge_attr[:, 0:2 * self.d] = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[1]]
        else:
            xy = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            if theta is None:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)

    def get_boundary(self):
        s = self.s
        n = self.n
        boundary1 = np.array(range(0, s))
        boundary2 = np.array(range(n - s, n))
        boundary3 = np.array(range(s, n, s))
        boundary4 = np.array(range(2 * s - 1, n, s))
        self.boundary = np.concatenate([boundary1, boundary2, boundary3, boundary4])

    def boundary_connectivity2d(self, stride=1):

        boundary = self.boundary[::stride]
        boundary_size = len(boundary)
        vertice1 = np.array(range(self.n))
        vertice1 = np.repeat(vertice1, boundary_size)
        vertice2 = np.tile(boundary, self.n)
        self.edge_index_boundary = np.stack([vertice2, vertice1], axis=0)
        self.n_edges_boundary = self.edge_index_boundary.shape[1]
        return torch.tensor(self.edge_index_boundary, dtype=torch.long)

    def attributes_boundary(self, f=None, theta=None):
        # if self.edge_index_boundary == None:
        #     self.boundary_connectivity2d()
        if f is None:
            if theta is None:
                edge_attr_boundary = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary, -1))
            else:
                edge_attr_boundary = np.zeros((self.n_edges_boundary, 3 * self.d))
                edge_attr_boundary[:, 0:2 * self.d] = self.grid[self.edge_index_boundary.T].reshape(
                    (self.n_edges_boundary, -1))
                edge_attr_boundary[:, 2 * self.d] = theta[self.edge_index_boundary[0]]
                edge_attr_boundary[:, 2 * self.d + 1] = theta[self.edge_index_boundary[1]]
        else:
            xy = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary, -1))
            if theta is None:
                edge_attr_boundary = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                edge_attr_boundary = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index_boundary[0]],
                                       theta[self.edge_index_boundary[1]])

        return torch.tensor(edge_attr_boundary, dtype=torch.float)


class RandomMeshGenerator(object):
    def __init__(self, real_space, mesh_size, sample_size):
        super(RandomMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.m = sample_size

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

        if self.m > self.n:
            self.m = self.n

        self.idx = np.array(range(self.n))
        self.grid_sample = self.grid

    def sample(self):
        perm = torch.randperm(self.n)
        self.idx = perm[:self.m]
        self.grid_sample = self.grid[self.idx]
        return self.idx

    def get_grid(self):
        return torch.tensor(self.grid_sample, dtype=torch.float)

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        rbf = np.exp(-pwd ** 2 / sigma ** 2)
        sample = np.random.binomial(1, rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            else:
                theta = theta[self.idx]
                edge_attr = np.zeros((self.n_edges, 3 * self.d))
                edge_attr[:, 0:2 * self.d] = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[1]]
        else:
            xy = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))
            if theta is None:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                theta = theta[self.idx]
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)


class RandomGridSplitter(object):
    def __init__(self, grid, resolution, m=200, l=2, radius=0.25):
        super(RandomGridSplitter, self).__init__()

        self.grid = grid
        self.resolution = resolution
        self.n = resolution ** 2
        self.m = m
        self.l = l
        self.radius = radius

        assert self.n % self.m == 0
        self.num = self.n // self.m

    def get_data(self, theta):

        data = []
        for i in range(self.l):
            perm = torch.randperm(self.n)
            perm = perm.reshape(self.num, self.m)

            for j in range(self.num):
                idx = perm[j, :].reshape(-1, )
                grid_sample = self.grid.reshape(self.n, -1)[idx]
                theta_sample = theta.reshape(self.n, -1)[idx]

                X = torch.cat([grid_sample, theta_sample], dim=1)

                pwd = sklearn.metrics.pairwise_distances(grid_sample)
                edge_index = np.vstack(np.where(pwd <= self.radius))
                n_edges = edge_index.shape[1]
                edge_index = torch.tensor(edge_index, dtype=torch.long)

                edge_attr = np.zeros((n_edges, 6))
                a = theta_sample[:, 0]
                edge_attr[:, :4] = grid_sample[edge_index.T].reshape(n_edges, -1)
                edge_attr[:, 4] = a[edge_index[0]]
                edge_attr[:, 5] = a[edge_index[1]]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)

                data.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, split_idx=idx))
        print('test', len(data), X.shape, edge_index.shape, edge_attr.shape)
        return data

    def assemble(self, pred, split_idx, batch_size2, sigma=1):
        assert len(pred) == len(split_idx)
        assert len(pred) == self.num * self.l // batch_size2

        out = torch.zeros(self.n, )
        for i in range(len(pred)):
            pred_i = pred[i].reshape(batch_size2, self.m)
            split_idx_i = split_idx[i].reshape(batch_size2, self.m)
            for j in range(batch_size2):
                pred_ij = pred_i[j, :].reshape(-1, )
                idx = split_idx_i[j, :].reshape(-1, )
                out[idx] = pred_ij

        out = out / self.l

        # out = gaussian_filter(out, sigma=sigma, mode='constant', cval=0)
        # out = torch.tensor(out, dtype=torch.float)
        return out.reshape(-1, )


class DownsampleGridSplitter(object):
    def __init__(self, grid, resolution, r, m=100, radius=0.15, edge_features=1):
        super(DownsampleGridSplitter, self).__init__()

        self.grid = grid.reshape(resolution, resolution, 2)
        # self.theta = theta.reshape(resolution, resolution,-1)
        # self.y = y.reshape(resolution, resolution,1)
        self.resolution = resolution
        if resolution % 2 == 1:
            self.s = int(((resolution - 1) / r) + 1)
        else:
            self.s = int(resolution / r)
        self.r = r
        self.n = resolution ** 2
        self.m = m
        self.radius = radius
        self.edge_features = edge_features

        self.index = torch.tensor(range(self.n), dtype=torch.long).reshape(self.resolution, self.resolution)

    def ball_connectivity(self, grid):
        pwd = sklearn.metrics.pairwise_distances(grid)
        edge_index = np.vstack(np.where(pwd <= self.radius))
        n_edges = edge_index.shape[1]
        return torch.tensor(edge_index, dtype=torch.long), n_edges

    def get_data(self, theta):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        data = []
        for x in range(self.r):
            for y in range(self.r):
                grid_sub = self.grid[x::self.r, y::self.r, :].reshape(-1, 2)
                theta_sub = theta[x::self.r, y::self.r, :].reshape(-1, theta_d)

                perm = torch.randperm(self.n)
                m = self.m - grid_sub.shape[0]
                idx = perm[:m]
                grid_sample = self.grid.reshape(self.n, -1)[idx]
                theta_sample = theta.reshape(self.n, -1)[idx]

                grid_split = torch.cat([grid_sub, grid_sample], dim=0)
                theta_split = torch.cat([theta_sub, theta_sample], dim=0)
                X = torch.cat([grid_split, theta_split], dim=1)

                edge_index, n_edges = self.ball_connectivity(grid_split)

                edge_attr = np.zeros((n_edges, 4 + self.edge_features * 2))
                a = theta_split[:, :self.edge_features]
                edge_attr[:, :4] = grid_split[edge_index.T].reshape(n_edges, -1)
                edge_attr[:, 4:4 + self.edge_features] = a[edge_index[0]]
                edge_attr[:, 4 + self.edge_features: 4 + self.edge_features * 2] = a[edge_index[1]]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)

                data.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx))
        print('test', len(data), X.shape, edge_index.shape, edge_attr.shape)
        return data

    def sample(self, theta, Y):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        Y = Y.reshape(self.resolution, self.resolution)

        x = torch.randint(0, self.r, (1,))
        y = torch.randint(0, self.r, (1,))

        grid_sub = self.grid[x::self.r, y::self.r, :].reshape(-1, 2)
        theta_sub = theta[x::self.r, y::self.r, :].reshape(-1, theta_d)
        Y_sub = Y[x::self.r, y::self.r].reshape(-1, )
        index_sub = self.index[x::self.r, y::self.r].reshape(-1, )
        n_sub = Y_sub.shape[0]

        if self.m >= n_sub:
            m = self.m - n_sub
            perm = torch.randperm(self.n)
            idx = perm[:m]
            grid_sample = self.grid.reshape(self.n, -1)[idx]
            theta_sample = theta.reshape(self.n, -1)[idx]
            Y_sample = Y.reshape(self.n, )[idx]

            grid_split = torch.cat([grid_sub, grid_sample], dim=0)
            theta_split = torch.cat([theta_sub, theta_sample], dim=0)
            Y_split = torch.cat([Y_sub, Y_sample], dim=0).reshape(-1, )
            index_split = torch.cat([index_sub, idx], dim=0).reshape(-1, )
            X = torch.cat([grid_split, theta_split], dim=1)

        else:
            grid_split = grid_sub
            theta_split = theta_sub
            Y_split = Y_sub.reshape(-1, )
            index_split = index_sub.reshape(-1, )
            X = torch.cat([grid_split, theta_split], dim=1)

        edge_index, n_edges = self.ball_connectivity(grid_split)

        edge_attr = np.zeros((n_edges, 4 + self.edge_features * 2))
        a = theta_split[:, :self.edge_features]
        edge_attr[:, :4] = grid_split[edge_index.T].reshape(n_edges, -1)
        edge_attr[:, 4:4 + self.edge_features] = a[edge_index[0]]
        edge_attr[:, 4 + self.edge_features: 4 + self.edge_features * 2] = a[edge_index[1]]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)
        data = Data(x=X, y=Y_split, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx,
                    sample_idx=index_split)
        print('train', X.shape, Y_split.shape, edge_index.shape, edge_attr.shape, index_split.shape)

        return data

    def assemble(self, pred, split_idx, batch_size2, sigma=1):
        assert len(pred) == len(split_idx)
        assert len(pred) == self.r ** 2 // batch_size2

        out = torch.zeros((self.resolution, self.resolution))
        for i in range(len(pred)):
            pred_i = pred[i].reshape(batch_size2, self.m)
            split_idx_i = split_idx[i]
            for j in range(batch_size2):
                pred_ij = pred_i[j, :]
                x, y = split_idx_i[j]
                if self.resolution % 2 == 1:
                    if x == 0:
                        nx = self.s
                    else:
                        nx = self.s - 1
                    if y == 0:
                        ny = self.s
                    else:
                        ny = self.s - 1
                else:
                    nx = self.s
                    ny = self.s
                # pred_ij = pred_i[idx : idx + nx * ny]
                out[x::self.r, y::self.r] = pred_ij[:nx * ny].reshape(nx, ny)

        out = gaussian_filter(out, sigma=sigma, mode='constant', cval=0)
        out = torch.tensor(out, dtype=torch.float)
        return out.reshape(-1, )


class TorusGridSplitter(object):
    def __init__(self, grid, resolution, r, m=100, radius=0.15, edge_features=1):
        super(TorusGridSplitter, self).__init__()

        self.grid = grid.reshape(resolution, resolution, 2)
        # self.theta = theta.reshape(resolution, resolution,-1)
        # self.y = y.reshape(resolution, resolution,1)
        self.resolution = resolution
        if resolution % 2 == 1:
            self.s = int(((resolution - 1) / r) + 1)
        else:
            self.s = int(resolution / r)
        self.r = r
        self.n = resolution ** 2
        self.m = m
        self.radius = radius
        self.edge_features = edge_features

        self.index = torch.tensor(range(self.n), dtype=torch.long).reshape(self.resolution, self.resolution)

    def pairwise_difference(self, grid1, grid2):
        n = grid1.shape[0]
        x1 = grid1[:, 0]
        y1 = grid1[:, 1]
        x2 = grid2[:, 0]
        y2 = grid2[:, 1]

        X1 = np.tile(x1.reshape(n, 1), [1, n])
        X2 = np.tile(x2.reshape(1, n), [n, 1])
        X_diff = X1 - X2
        Y1 = np.tile(y1.reshape(n, 1), [1, n])
        Y2 = np.tile(y2.reshape(1, n), [n, 1])
        Y_diff = Y1 - Y2

        return X_diff, Y_diff

    def torus_connectivity(self, grid):
        pwd0 = sklearn.metrics.pairwise_distances(grid, grid)
        X_diff0, Y_diff0 = self.pairwise_difference(grid, grid)

        grid1 = grid
        grid1[:, 0] = grid[:, 0] + 1
        pwd1 = sklearn.metrics.pairwise_distances(grid, grid1)
        X_diff1, Y_diff1 = self.pairwise_difference(grid, grid1)

        grid2 = grid
        grid2[:, 1] = grid[:, 1] + 1
        pwd2 = sklearn.metrics.pairwise_distances(grid, grid2)
        X_diff2, Y_diff2 = self.pairwise_difference(grid, grid2)

        grid3 = grid
        grid3[:, :] = grid[:, :] + 1
        pwd3 = sklearn.metrics.pairwise_distances(grid, grid3)
        X_diff3, Y_diff3 = self.pairwise_difference(grid, grid3)

        grid4 = grid
        grid4[:, 0] = grid[:, 0] + 1
        grid4[:, 1] = grid[:, 1] - 1
        pwd4 = sklearn.metrics.pairwise_distances(grid, grid4)
        X_diff4, Y_diff4 = self.pairwise_difference(grid, grid4)

        PWD = np.stack([pwd0, pwd1, pwd2, pwd3, pwd4], axis=2)
        X_DIFF = np.stack([X_diff0, X_diff1, X_diff2, X_diff3, X_diff4], axis=2)
        Y_DIFF = np.stack([Y_diff0, Y_diff1, Y_diff2, Y_diff3, Y_diff4], axis=2)
        pwd = np.min(PWD, axis=2)
        pwd_index = np.argmin(PWD, axis=2)
        edge_index = np.vstack(np.where(pwd <= self.radius))
        pwd_index = pwd_index[np.where(pwd <= self.radius)]
        PWD_index = (np.where(pwd <= self.radius)[0], np.where(pwd <= self.radius)[1], pwd_index)
        distance = PWD[PWD_index]
        X_difference = X_DIFF[PWD_index]
        Y_difference = Y_DIFF[PWD_index]
        n_edges = edge_index.shape[1]
        return torch.tensor(edge_index, dtype=torch.long), n_edges, distance, X_difference, Y_difference

    def get_data(self, theta):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        data = []
        for x in range(self.r):
            for y in range(self.r):
                grid_sub = self.grid[x::self.r, y::self.r, :].reshape(-1, 2)
                theta_sub = theta[x::self.r, y::self.r, :].reshape(-1, theta_d)

                perm = torch.randperm(self.n)
                m = self.m - grid_sub.shape[0]
                idx = perm[:m]
                grid_sample = self.grid.reshape(self.n, -1)[idx]
                theta_sample = theta.reshape(self.n, -1)[idx]

                grid_split = torch.cat([grid_sub, grid_sample], dim=0)
                theta_split = torch.cat([theta_sub, theta_sample], dim=0)
                X = torch.cat([grid_split, theta_split], dim=1)

                edge_index, n_edges, distance, X_difference, Y_difference = self.torus_connectivity(grid_split)

                edge_attr = np.zeros((n_edges, 3 + self.edge_features * 2))
                a = theta_split[:, :self.edge_features]
                edge_attr[:, 0] = X_difference.reshape(n_edges, )
                edge_attr[:, 1] = Y_difference.reshape(n_edges, )
                edge_attr[:, 2] = distance.reshape(n_edges, )
                edge_attr[:, 3:3 + self.edge_features] = a[edge_index[0]]
                edge_attr[:, 3 + self.edge_features: 4 + self.edge_features * 2] = a[edge_index[1]]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)

                data.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx))
        print('test', len(data), X.shape, edge_index.shape, edge_attr.shape)
        return data

    def sample(self, theta, Y, connectivity='ball'):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        Y = Y.reshape(self.resolution, self.resolution)

        x = torch.randint(0, self.r, (1,))
        y = torch.randint(0, self.r, (1,))

        grid_sub = self.grid[x::self.r, y::self.r, :].reshape(-1, 2)
        theta_sub = theta[x::self.r, y::self.r, :].reshape(-1, theta_d)
        Y_sub = Y[x::self.r, y::self.r].reshape(-1, )
        index_sub = self.index[x::self.r, y::self.r].reshape(-1, )
        n_sub = Y_sub.shape[0]

        if self.m >= n_sub:
            m = self.m - n_sub
            perm = torch.randperm(self.n)
            idx = perm[:m]
            grid_sample = self.grid.reshape(self.n, -1)[idx]
            theta_sample = theta.reshape(self.n, -1)[idx]
            Y_sample = Y.reshape(self.n, )[idx]

            grid_split = torch.cat([grid_sub, grid_sample], dim=0)
            theta_split = torch.cat([theta_sub, theta_sample], dim=0)
            Y_split = torch.cat([Y_sub, Y_sample], dim=0).reshape(-1, )
            index_split = torch.cat([index_sub, idx], dim=0).reshape(-1, )
            X = torch.cat([grid_split, theta_split], dim=1)

        else:
            grid_split = grid_sub
            theta_split = theta_sub
            Y_split = Y_sub.reshape(-1, )
            index_split = index_sub.reshape(-1, )
            X = torch.cat([grid_split, theta_split], dim=1)

        edge_index, n_edges, distance, X_difference, Y_difference = self.torus_connectivity(grid_split)

        edge_attr = np.zeros((n_edges, 3 + self.edge_features * 2))
        a = theta_split[:, :self.edge_features]
        edge_attr[:, 0] = X_difference.reshape(n_edges, )
        edge_attr[:, 1] = Y_difference.reshape(n_edges, )
        edge_attr[:, 2] = distance.reshape(n_edges, )
        edge_attr[:, 3:3 + self.edge_features] = a[edge_index[0]]
        edge_attr[:, 3 + self.edge_features: 4 + self.edge_features * 2] = a[edge_index[1]]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)
        data = Data(x=X, y=Y_split, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx,
                    sample_idx=index_split)
        print('train', X.shape, Y_split.shape, edge_index.shape, edge_attr.shape, index_split.shape)

        return data

    def assemble(self, pred, split_idx, batch_size2, sigma=1):
        assert len(pred) == len(split_idx)
        assert len(pred) == self.r ** 2 // batch_size2

        out = torch.zeros((self.resolution, self.resolution))
        for i in range(len(pred)):
            pred_i = pred[i].reshape(batch_size2, self.m)
            split_idx_i = split_idx[i]
            for j in range(batch_size2):
                pred_ij = pred_i[j, :]
                x, y = split_idx_i[j]
                if self.resolution % 2 == 1:
                    if x == 0:
                        nx = self.s
                    else:
                        nx = self.s - 1
                    if y == 0:
                        ny = self.s
                    else:
                        ny = self.s - 1
                else:
                    nx = self.s
                    ny = self.s
                # pred_ij = pred_i[idx : idx + nx * ny]
                out[x::self.r, y::self.r] = pred_ij[:nx * ny].reshape(nx, ny)

        out = gaussian_filter(out, sigma=sigma, mode='constant', cval=0)
        out = torch.tensor(out, dtype=torch.float)
        return out.reshape(-1, )


def downsample(data, grid_size, l):
    data = data.reshape(-1, grid_size, grid_size)
    data = data[:, ::l, ::l]
    data = data.reshape(-1, (grid_size // l) ** 2)
    return data


def grid(n_x, n_y):
    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)
    # xs = np.array(range(n_x))
    # ys = np.array(range(n_y))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []
    for y in range(n_y):
        for x in range(n_x):
            i = y * n_x + x
            if (x != n_x - 1):
                edge_index.append((i, i + 1))
                edge_attr.append((1, 0, 0))
                edge_index.append((i + 1, i))
                edge_attr.append((-1, 0, 0))

            if (y != n_y - 1):
                edge_index.append((i, i + n_x))
                edge_attr.append((0, 1, 0))
                edge_index.append((i + n_x, i))
                edge_attr.append((0, -1, 0))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


def grid_edge(n_x, n_y, a):
    a = a.reshape(n_x, n_y)
    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)
    # xs = np.array(range(n_x))
    # ys = np.array(range(n_y))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []
    for y in range(n_y):
        for x in range(n_x):
            i = y * n_x + x
            if (x != n_x - 1):
                d = 1 / n_x
                a1 = a[x, y]
                a2 = a[x + 1, y]
                edge_index.append((i, i + 1))
                edge_attr.append((d, a1, a2))
                edge_index.append((i + 1, i))
                edge_attr.append((d, a2, a1))

            if (y != n_y - 1):
                d = 1 / n_y
                a1 = a[x, y]
                a2 = a[x, y + 1]
                edge_index.append((i, i + n_x))
                edge_attr.append((d, a1, a2))
                edge_index.append((i + n_x, i))
                edge_attr.append((d, a2, a1))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


def grid_edge_aug(n_x, n_y, a):
    a = a.reshape(n_x, n_y)
    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)
    # xs = np.array(range(n_x))
    # ys = np.array(range(n_y))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []
    for y in range(n_y):
        for x in range(n_x):
            i = y * n_x + x
            if (x != n_x - 1):
                d = 1 / n_x
                a1 = a[x, y]
                a2 = a[x + 1, y]
                edge_index.append((i, i + 1))
                edge_attr.append((d, a1, a2, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))
                edge_index.append((i + 1, i))
                edge_attr.append((d, a2, a1, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))

            if (y != n_y - 1):
                d = 1 / n_y
                a1 = a[x, y]
                a2 = a[x, y + 1]
                edge_index.append((i, i + n_x))
                edge_attr.append((d, a1, a2, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))
                edge_index.append((i + n_x, i))
                edge_attr.append((d, a2, a1, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


def grid_edge_aug_full(n_x, n_y, r, a):
    n = n_x * n_y

    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)

    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []

    for i1 in range(n):
        x1 = grid[i1]
        for i2 in range(n):
            x2 = grid[i2]

            d = np.linalg.norm(x1 - x2)

            if (d <= r):
                a1 = a[i1]
                a2 = a[i2]
                edge_index.append((i1, i2))
                edge_attr.append((d, a1, a2, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))
                edge_index.append((i2, i1))
                edge_attr.append((d, a2, a1, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


def multi_grid(depth, n_x, n_y, grid, params):
    edge_index_global = []
    edge_attr_global = []
    X_global = []
    num_nodes = 0

    # build connected graph
    for l in range(depth):
        h_x_l = n_x // (2 ** l)
        h_y_l = n_y // (2 ** l)
        n_l = h_x_l * h_y_l

        a = downsample(params, n_x, (2 ** l))
        if grid == 'grid':
            X, edge_index_inner, edge_attr_inner = grid(h_y_l, h_x_l)
        elif grid == 'grid_edge':
            X, edge_index_inner, edge_attr_inner = grid_edge(h_y_l, h_x_l, a)
        elif grid == 'grid_edge_aug':
            X, edge_index_inner, edge_attr_inner = grid_edge(h_y_l, h_x_l, a)

        # update index
        edge_index_inner = edge_index_inner + num_nodes
        edge_index_global.append(edge_index_inner)
        edge_attr_global.append(edge_attr_inner)

        # construct X
        # if (is_high):
        #     X = torch.cat([torch.zeros(n_l, l * 2), X, torch.zeros(n_l, (depth - 1 - l) * 2)], dim=1)
        # else:
        #     X_l = torch.tensor(l, dtype=torch.float).repeat(n_l, 1)
        #     X = torch.cat([X, X_l], dim=1)
        X_global.append(X)

        # construct edges
        index1 = torch.tensor(range(n_l), dtype=torch.long)
        index1 = index1 + num_nodes
        num_nodes += n_l

        # #construct inter-graph edge
        if l != depth - 1:
            index2 = np.array(range(n_l // 4)).reshape(h_x_l // 2, h_y_l // 2)  # torch.repeat is different from numpy
            index2 = index2.repeat(2, axis=0).repeat(2, axis=1)
            index2 = torch.tensor(index2).reshape(-1)
            index2 = index2 + num_nodes
            index2 = torch.tensor(index2, dtype=torch.long)

            edge_index_inter1 = torch.cat([index1, index2], dim=-1).reshape(2, -1)
            edge_index_inter2 = torch.cat([index2, index1], dim=-1).reshape(2, -1)
            edge_index_inter = torch.cat([edge_index_inter1, edge_index_inter2], dim=1)

            edge_attr_inter1 = torch.tensor((0, 0, 1), dtype=torch.float).repeat(n_l, 1)
            edge_attr_inter2 = torch.tensor((0, 0, -1), dtype=torch.float).repeat(n_l, 1)
            edge_attr_inter = torch.cat([edge_attr_inter1, edge_attr_inter2], dim=0)

            edge_index_global.append(edge_index_inter)
            edge_attr_global.append(edge_attr_inter)

    X = torch.cat(X_global, dim=0)
    edge_index = torch.cat(edge_index_global, dim=1)
    edge_attr = torch.cat(edge_attr_global, dim=0)
    mask_index = torch.tensor(range(n_x * n_y), dtype=torch.long)
    # print('create multi_grid with size:', X.shape,  edge_index.shape, edge_attr.shape, mask_index.shape)

    return (X, edge_index, edge_attr, mask_index, num_nodes)
