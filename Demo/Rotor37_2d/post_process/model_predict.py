import torch
import os
import numpy as np
from post_process.post_data import Post_2d
from run_FNO import feature_transform
from Demo.Rotor37_2d.utilizes_rotor37 import get_grid
class DLModelPost(object):
    def __init__(self, netmodel, Device,
                 name=None,
                 in_norm=None,
                 out_norm=None,
                 grid_size=64,
                 ):
        self.netmodel = netmodel
        self.Device = Device
        self.name = name

        self.in_norm = in_norm
        self.out_norm = out_norm
        self.grid_size = grid_size

    def predicter_2d(self, input, input_norm=False):
        """
        加载完整的模型预测输入的坐标
        Net_model 训练完成的模型
        input 模型的输入 shape:[num, input_dim]
        """
        if len(input.shape)==1:
            input = input[np.newaxis, :]
        if not input_norm:  # 如果没有归一化，需要将输入归一化
            input = self.in_norm.norm(input)
        input = torch.tensor(input, dtype=torch.float)
        input = input.to(self.Device)
        self.netmodel.eval()

        if self.name in ("FNO", "UNet", "Transformer"):
            input = torch.tensor(np.tile(input[:, None, None, :], (1, self.grid_size, self.grid_size, 1)), dtype=torch.float)
            grid = feature_transform(input)
            pred = self.netmodel(input, grid)
        else:
            pred = self.netmodel(input)

        pred = pred.reshape([pred.shape[0], self.grid_size, self.grid_size, -1])
        pred = self.out_norm.back(pred)

        return pred.detach().numpy()

    def predictor_value(self, input, input_para=None, parameterList=None, input_norm=False):
        if not isinstance(parameterList, list):
            parameterList = [parameterList]

        pred_2d = self.predicter_2d(input, input_norm=input_norm)
        if input_para is None:
            input_para = {
                "PressureStatic": 0,
                "TemperatureStatic": 1,
                "Density": 2,
                "PressureTotalW": 3,
                "TemperatureTotalW": 4,
            }

        grid = get_grid(real_path=os.path.join("..", "data"))
        post_pred = Post_2d(pred_2d, grid,
                            inputDict=input_para,
                            )

        Rst = []
        for parameter_Name in parameterList:
            value = getattr(post_pred, parameter_Name)
            value = post_pred.span_density_average(value[..., -1])
            Rst.append(value)

        return np.concatenate(Rst, axis=1)