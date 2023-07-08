import torch
import torch.nn as nn


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class DeepModel_single(nn.Module):
    def __init__(self, planes, dropout=0.0):
        super(DeepModel_single, self).__init__()
        self.planes = planes
        self.layers = []
        for i in range(len(self.planes) - 2):
            self.layers.append(nn.Linear(self.planes[i], self.planes[i + 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(self.planes[-2], self.planes[-1]))
        self.Linear = nn.Sequential(*self.layers)


    def forward(self, x):
        y = self.Linear(x)
        return y


class DeepModel_multi(nn.Module):
    def __init__(self, planes, dropout=0.0):
        super(DeepModel_multi, self).__init__()
        self.planes = planes

        self.layers = nn.ModuleList()
        for j in range(self.planes[-1]):
            layer = []
            for i in range(len(self.planes) - 2):
                layer.append(nn.Linear(self.planes[i], self.planes[i + 1]))
                layer.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout))
            layer.append(nn.Linear(self.planes[-2], 1))
            self.layers.append(nn.Sequential(*layer))


    def forward(self, x):

        y = []
        for i in range(self.planes[-1]):
            y.append(self.layers[i](x))

        return torch.cat(y, dim=-1)





