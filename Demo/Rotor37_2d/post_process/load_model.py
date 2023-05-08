import torch
import os
import numpy as np

from post_process.post_data import Post_2d
from Demo.Rotor37_2d.utilizes_rotor37 import get_grid, get_origin
from Utilizes.process_data import DataNormer
import yaml





def loaddata(name,ntrain,nvalid):
    batch_size = 128
    design, fields = get_origin(realpath=os.path.join("..", "data"))  # 获取原始数据
    if name in ("FNO", "UNet", "Transformer"):
        input = np.tile(design[:, None, None, :], (1, 64, 64, 1))
    else:
        input = design

    input = torch.tensor(input, dtype=torch.float)

    output = fields
    output = torch.tensor(output, dtype=torch.float)
    print(input.shape, output.shape)

    train_x = input[:ntrain, :]
    train_y = output[:ntrain, :]
    valid_x = input[ntrain:ntrain + nvalid, :]
    valid_y = output[ntrain:ntrain + nvalid, :]

    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)

    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)

    train_y = train_y.reshape([train_x.shape[0], -1])
    valid_y = valid_y.reshape([valid_x.shape[0], -1])

    if name in ("deepONet"):
        grid = get_grid()
        grid_trans = torch.tensor(grid[np.newaxis, :, :, :], dtype=torch.float)
        train_grid = torch.tile(grid_trans, [train_x.shape[0], 1, 1, 1])  # 所有样本的坐标是一致的。
        valid_grid = torch.tile(grid_trans, [valid_x.shape[0], 1, 1, 1])

        grid_normalizer = DataNormer(train_grid.numpy(), method='mean-std')  # 这里的axis不一样了
        train_grid = grid_normalizer.norm(train_grid)
        valid_grid = grid_normalizer.norm(valid_grid)

        # grid_trans = grid_trans.reshape([1, -1, 2])
        train_grid = train_grid.reshape([train_x.shape[0], -1, 2])
        valid_grid = valid_grid.reshape([valid_x.shape[0], -1, 2])

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_grid, train_y),
                                                   batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_grid, valid_y),
                                                   batch_size=batch_size, shuffle=False, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                                   batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                                   batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, valid_loader, x_normalizer, y_normalizer

def rebuild_model(work_path, Device, in_dim=28, out_dim=5, name=None, mode=10):
    """
    rebuild the model with pth files
    """
    # rebuild the model
    if 'MLP' in name:
        from run_MLP import MLP
        from run_MLP import inference
        layer_mat = [in_dim, 256, 256, 256, 256, 256, 256, 256, 256, out_dim * 64 * 64]
        Net_model = MLP(layer_mat, is_BatchNorm=False).to(Device)
    elif 'deepONet' in name:
        from don.DeepONets import DeepONetMulti
        from run_deepONet import inference
        Net_model = DeepONetMulti(input_dim=2, operator_dims=[28, ], output_dim=5,
                                  planes_branch=[128] * 5, planes_trunk=[128] * 5).to(Device)
    elif 'FNO' in name:
        from fno.FNOs import FNO2d
        from run_FNO import inference
        Net_model = FNO2d(in_dim=in_dim, out_dim=out_dim, modes=mode, width=64, depth=4, steps=1,
                          padding=8, activation='gelu').to(Device)
    elif 'UNet' in name:
        from cnn.ConvNets import UNet2d
        from run_UNet import inference
        Net_model = UNet2d(in_sizes=[64, 64, 28], out_sizes=[64, 64, 5], width=64,
                           depth=4, steps=1, activation='gelu', dropout=0).to(Device)
    elif 'Transformer' in name:
        from transformer.Transformers import SimpleTransformer, FourierTransformer2D
        from run_Trans import inference
        with open(os.path.join('transformer_config.yml')) as f:
            config = yaml.full_load(f)
            config = config['Rotor37_2d']
            config['fourier_modes'] = mode
        # 建立网络
        Net_model = FourierTransformer2D(**config).to(Device)

    isExist = os.path.exists(os.path.join(work_path, 'latest_model.pth'))
    if isExist:
        checkpoint = torch.load(os.path.join(work_path, 'latest_model.pth'), map_location=Device)
        Net_model.load_state_dict(checkpoint['net_model'])
        return Net_model, inference
    else:
        print("The pth file is not exist, CHECK PLEASE!")
        return None, None

def get_true_pred(loader, Net_model, inference, Device,
                  name='MLP', out_dim=5):
    if name in ('MLP'):
        grid, true, pred = inference(loader, Net_model, Device)
    else:
        coord, grid, true, pred = inference(loader, Net_model, Device)
    true = true.reshape([true.shape[0], 64, 64, out_dim])
    pred = pred.reshape([pred.shape[0], 64, 64, out_dim])

    return true, pred

if __name__ == "__main__":
    #建立模型并读入参数
    name = 'MLP'
    work_path = os.path.join('work', name)

    in_dim = 28
    out_dim = 5

    layer_mat = [in_dim, 256, 256, 256, 256, 256, 256, 256, 256, out_dim * 64 * 64]
    Net_model = MLP(layer_mat, is_BatchNorm=False)

    checkpoint = torch.load(os.path.join(work_path, 'latest_model.pth'))
    Net_model.load_state_dict(checkpoint['net_model'])

    #输出预测结果