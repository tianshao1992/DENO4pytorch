import torch
import os
from post_data import Post_2d
from run_MLP import MLP, get_grid, get_origin
from run_FNO import inference
from Utilizes.process_data import DataNormer
from Utilizes.visual_data import MatplotlibVision
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from fno.FNOs import FNO2d
from cnn.ConvNets import UNet2d

if __name__ == "__main__":
    in_dim = 28
    out_dim = 5

    ntrain = 800
    nvalid = 300

    #=====================================#
    #MLP
    #=====================================#
    # batch_size = 32
    # epochs = 1000
    # learning_rate = 0.001
    # scheduler_step = 800
    # scheduler_gamma = 0.1

    # =====================================#
    # FNO&UNet
    # =====================================#

    # modes = (12, 12)
    # width = 64
    # depth = 4
    # steps = 1
    # padding = 8
    # dropout = 0.0
    #
    # batch_size = 32
    # epochs = 1000
    # learning_rate = 0.001
    # scheduler_step = 800
    # scheduler_gamma = 0.1

    # =====================================#
    # UNet
    # =====================================#

    modes = (12, 12)
    width = 64
    depth = 4
    steps = 1
    padding = 8
    dropout = 0.0

    batch_size = 32
    epochs = 1000
    learning_rate = 0.001
    scheduler_step = 800
    scheduler_gamma = 0.1


    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    ################################################################
    # load data
    ################################################################

    design, fields = get_origin()  # 获取原始数据

    # input = design
    input = np.tile(design[:, None, None, :], (1, 64, 64, 1))
    input = torch.tensor(input, dtype=torch.float)

    output = fields[:, 0, :, :, :].transpose((0, 2, 3, 1))
    # output = output.reshape([output.shape[0],-1])
    output = torch.tensor(output, dtype=torch.float)
    print(input.shape, output.shape)

    r1 = 1
    r2 = 1
    s1 = int(((64 - 1) / r1) + 1)
    s2 = int(((64 - 1) / r2) + 1)

    train_x = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
    train_y = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    valid_x = input[ntrain:ntrain + nvalid, ::r1, ::r2][:, :s1, :s2]
    valid_y = output[ntrain:ntrain + nvalid, ::r1, ::r2][:, :s1, :s2]

    # train_x = input[:ntrain, :]
    # train_y = output[:ntrain, :]
    # valid_x = input[ntrain:ntrain + nvalid, :]
    # valid_y = output[ntrain:ntrain + nvalid, :]

    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)

    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)

    train_y = train_y.reshape([train_x.shape[0], -1])
    valid_y = valid_y.reshape([valid_x.shape[0], -1])

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                               batch_size=batch_size, shuffle=False, drop_last=True)


    ################################################################
    # Netmodel
    ################################################################
    name = 'UNet'
    work_path = os.path.join('../work', name)

    if name == 'FNO':
        Net_model = FNO2d(in_dim=in_dim, out_dim=out_dim, modes=modes, width=width, depth=depth, steps=steps,
                          padding=padding, activation='gelu').to(Device)
    elif name == 'UNet':
        Net_model = UNet2d(in_sizes=train_x.shape[1:], out_sizes=train_y.shape[1:], width=width,
                           depth=depth, steps=steps, activation='gelu', dropout=dropout).to(Device)
    elif name == 'MLP':
        layer_mat = [in_dim, 256, 256, 256, 256, 256, 256, 256, 256, out_dim * 64 * 64]
        Net_model = MLP(layer_mat, is_BatchNorm=False).to(Device)

    checkpoint = torch.load(os.path.join(work_path, 'latest_model.pth'))
    Net_model.load_state_dict(checkpoint['net_model'])

    ################################################################
    # Predict and Compare
    ################################################################

    # train_coord, train_true, train_pred = inference(train_loader, Net_model, Device)
    # valid_coord, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

    train_coord, train_grid, train_true, train_pred = inference(train_loader, Net_model, Device)
    valid_coord, valid_grid, valid_true, valid_pred = inference(valid_loader, Net_model, Device)

    train_true = train_true.reshape([train_true.shape[0], 64, 64, out_dim])
    train_pred = train_pred.reshape([train_pred.shape[0], 64, 64, out_dim])
    valid_true = valid_true.reshape([valid_true.shape[0], 64, 64, out_dim])
    valid_pred = valid_pred.reshape([valid_pred.shape[0], 64, 64, out_dim])

    train_true = y_normalizer.back(train_true)
    train_pred = y_normalizer.back(train_pred)
    valid_true = y_normalizer.back(valid_true)
    valid_pred = y_normalizer.back(valid_pred)

    grid = get_grid()
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('p', 't', 'rho', 'alf', 'v'))

    for fig_id in range(10):

        post_true = Post_2d(train_true[fig_id],grid)
        post_pred = Post_2d(train_pred[fig_id], grid)
        # plt.plot(post_true.Efficiency[:,-1],np.arange(64),label="true")
        # plt.plot(post_pred.Efficiency[:, -1], np.arange(64), label="pred")
        fig, axs = plt.subplots(1, 1, figsize=(10, 5), num=1)
        Visual.plot_value(fig, axs ,post_true.Efficiency[:,-1], np.arange(64), label="true")
        Visual.plot_value(fig, axs, post_pred.Efficiency[:, -1], np.arange(64), label="pred",title="train_solution",xylabels=("efficiency","span"))
        fig.savefig(os.path.join(work_path, 'train_solution_eff_' + str(fig_id) + '.jpg'))
        plt.close(fig)

    for fig_id in range(10):

        post_true = Post_2d(valid_true[fig_id],grid)
        post_pred = Post_2d(valid_pred[fig_id], grid)
        fig, axs = plt.subplots(1, 1, figsize=(10, 5), num=1)
        Visual.plot_value(fig, axs ,post_true.Efficiency[:,-1], np.arange(64), label="true")
        Visual.plot_value(fig, axs, post_pred.Efficiency[:, -1], np.arange(64), label="pred",title="train_solution",xylabels=("efficiency","span"))
        fig.savefig(os.path.join(work_path, 'valid_solution_eff_' + str(fig_id) + '.jpg'))
        plt.close(fig)


