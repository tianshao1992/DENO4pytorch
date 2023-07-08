import torch
import numpy as np



def read_data():
    all = torch.load('data\\ANN_span0.5_2023.pth', 'r')

    ind = all['ind']
    # nodes = all['nodes'][:, :2]
    nodes = all['nodes']
    mdess = all['design'][:, :11]
    nodes_ = np.concatenate(nodes, axis=0)

    c_min_max = np.stack([np.min(nodes_[:, :2], axis=0), np.max(nodes_[:, :2], axis=0)], axis=0)
    f_min_max = np.stack([np.min(nodes_[:, 2:], axis=0), np.max(nodes_[:, 2:], axis=0)], axis=0)
    d_min_max = np.stack([np.min(mdess, axis=0), np.max(mdess, axis=0)], axis=0)


    c_norm = data_norm(c_min_max)
    f_norm = data_norm(f_min_max)
    d_norm = data_norm(d_min_max)


    return nodes, mdess, (d_norm, c_norm, f_norm)




class data_norm():

    def __init__(self, data, method="min-max"):
        axis = tuple(range(len(data.shape) - 1))
        self.method = method

        if method == "min-max":
            self.max = np.max(data, axis=axis)
            self.min = np.min(data, axis=axis)

        elif method == "mean-std":
            self.mean = np.mean(data, axis=axis)
            self.std = np.std(data, axis=axis)


    def norm(self, x):
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = 2 * (x - torch.tensor(self.min, device=x.device)) \
                    / (torch.tensor(self.max, device=x.device) - torch.tensor(self.min, device=x.device) + 1e-10) - 1
            elif self.method == "mean-std":
                x = (x - torch.tensor(self.mean, device=x.device)) / (torch.tensor(self.std, device=x.device) + 1e-10)
        else:
            if self.method == "min-max":
                x = 2 * (x - self.min) / (self.max - self.min+1e-10) - 1
            elif self.method == "mean-std":
                x = (x - self.mean) / (self.std + 1e-10)

        return x

    def back(self, x):
        if torch.is_tensor(x):
            if self.method == "min-max":
                x = (x + 1) / 2 * (torch.tensor(self.max, device=x.device)
                                   - torch.tensor(self.min, device=x.device) + 1e-10) + torch.tensor(self.min, device=x.device)
            elif self.method == "mean-std":
                x = x * (torch.tensor(self.std, device=x.device) + 1e-10) + torch.tensor(self.mean, device=x.device)
        else:
            if self.method == "min-max":
                x = (x + 1) / 2 * (self.max - self.min+1e-10) + self.min
            elif self.method == "mean-std":
                x = x * (self.std + 1e-10) + self.mean
        return x


class custom_dataset(object):
    def __init__(self, nodes, design, data_norm, get_size=4000):
        self.design = design
        self.nodes = nodes
        self.get_size = get_size

        self.design_norm = data_norm[0]
        self.coordn_norm = data_norm[1]
        self.fields_norm = data_norm[2]

    def __getitem__(self, index):  # 根据 idx 取出其中一个
        coords = self.nodes[index][:, :2]
        fields = self.nodes[index][:, 2:]

        design = self.design[index]

        nodes_num = coords.shape[0]
        ind = np.concatenate([np.random.permutation(nodes_num)] * 3, axis=0)[:self.get_size]
        fields = fields[ind]
        coords = coords[ind]

        design = self.design_norm.norm(design)
        fields = self.fields_norm.norm(fields)
        coords = self.coordn_norm.norm(coords)

        design = design * np.ones((fields.shape[0], 1), dtype=np.float32)

        return torch.tensor(np.concatenate((coords, design), axis=-1), dtype=torch.float32), \
               torch.tensor(fields, dtype=torch.float32)

    def get_regular(self, index):

        coords = self.nodes[index][:, :2]
        fields = self.nodes[index][:, 2:]
        design = self.design[index]

        design = self.design_norm.norm(design)
        fields = self.fields_norm.norm(fields)
        coords = self.coordn_norm.norm(coords)

        design = design * np.ones((fields.shape[0], 1), dtype=np.float32)

        return torch.tensor(np.concatenate((coords, design), axis=-1), dtype=torch.float32), \
               torch.tensor(fields, dtype=torch.float32)


    def __len__(self):
        return len(self.nodes)


if __name__ == '__main__':
    read_data()