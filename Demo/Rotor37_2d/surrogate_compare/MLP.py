import numpy as np
import torch
from collections import OrderedDict
import os
from post_process.load_model import get_noise

class MLP(torch.nn.Module):
    def __init__(self, layers=None, is_BatchNorm=True,
                 in_dim=None,
                 out_dim=None,
                 n_hidden=None,
                 num_layers=None):
        if layers is None:
            layers = [in_dim]
            for ii in range(num_layers-2):
                layers.append(n_hidden)
            layers.append(out_dim)
        super(MLP, self).__init__()
        self.depth = len(layers)
        self.activation = torch.nn.GELU
        #先写完整的layerslist
        layer_list = []
        for i in range(self.depth-2):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1])))
            if is_BatchNorm is True:
                layer_list.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))

        #最后一层，输出层
        layer_list.append(('layer_%d' % (self.depth-2), torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)
        #再直接使用sequential生成网络
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self,x):
        y = self.layers(x)
        return y

def get_pred(npz_path, n_train, n_noise, parameter, Device):

    data = np.load(npz_path)
    design = torch.tensor(data["Design"], dtype=torch.float32)

    value = torch.tensor(data[parameter].squeeze(), dtype=torch.float32)

    train_x = design[:n_train]
    train_y = value[:n_train]
    noise = get_noise(train_y.numpy().shape, n_noise) * np.mean(train_y.numpy())
    train_y = train_y + torch.tensor(noise, dtype=torch.float32)

    valid_x = design[-400:]
    valid_y = value[-400:]

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                               batch_size=32, shuffle=True, drop_last=True)
    in_dim = 28
    out_dim = 1
    # 建立网络
    layer_mat = [in_dim, 32, 32, 32, out_dim]
    Net_model = MLP(layers=layer_mat, is_BatchNorm=False)
    Net_model = Net_model.to(Device)

    Loss_func = torch.nn.MSELoss()
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=0.01, betas=(0.7, 0.9), weight_decay=1e-4)
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=400, gamma=0.1)

    Net_model.train()
    for epoch in range(500):
        loss = train(train_loader, Net_model, Loss_func, Optimizer, Scheduler, Device)
        if epoch > 0 and epoch % 10 == 0:
            print("epoch: {},  loss: {}".format(epoch, loss))

    Net_model.eval()
    with torch.no_grad():
        pred = Net_model(valid_x.to(Device))
    pred = pred.cpu().numpy()
    valid_y = valid_y.cpu().numpy()
    valid_y = valid_y[:, np.newaxis]

    test_results = np.mean(np.abs((valid_y - pred) / valid_y), axis=0)
    print(parameter + "_error = " + str(test_results))

    return pred.squeeze()

def train(dataloader, netmodel, lossfunc, optimizer, scheduler, device):
    train_loss = 0
    for batch, (input,output) in enumerate(dataloader):
        input = input.to(device)
        output = output.to(device)
        pred = netmodel(input)
        loss = lossfunc(pred, output)
        optimizer.zero_grad()
        loss.backward() # 自动微分
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()

    return train_loss / (batch + 1)

if __name__ == "__main__":
    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    npz_path = os.path.join("..", "data", "surrogate_data", "scalar_value.npz")
    # dict_all = {}
    # data = np.load(npz_path)
    # design = torch.tensor(data["Design"], dtype=torch.float32)

    dict_num = {}
    dict_noise = {}

    parameterList = [
        "PressureRatioV", "TemperatureRatioV",
        "Efficiency", "EfficiencyPoly",
        "PressureLossR", "EntropyStatic",
        "MachIsentropic", "Load",
        "MassFlow"]
    for parameter in parameterList:

        n_trainList = [500, 1000, 1500, 2000, 2500]
        n_noiseList = [0, 0.005, 0.01, 0.05, 0.1]

        data_box_num = np.zeros([400, 5])
        for ii, n_train in enumerate(n_trainList):
            pred = get_pred(npz_path, n_train, 0, parameter, Device)
            data_box_num[:, ii] = pred.copy()
        dict_num.update({parameter: data_box_num})

        data_box_noise = np.zeros([400, 5])
        for ii, n_noise in enumerate(n_noiseList):
            pred = get_pred(npz_path, 2500, n_noise, parameter, Device)
            data_box_noise[:, ii] = pred.copy()
        dict_noise.update({parameter: data_box_noise})

        np.savez(os.path.join("..", "data", "surrogate_data", "ANN_num.npz"), **dict_num)
        np.savez(os.path.join("..", "data", "surrogate_data", "ANN_noise.npz"), **dict_noise)









