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

        self.netmodel.eval()

        if self.name in ("FNO", "UNet", "Transformer"):
            input = torch.tensor(np.tile(input[:, None, None, :], (1, self.grid_size, self.grid_size, 1)), dtype=torch.float)
            input = input.to(self.Device)
            grid = feature_transform(input)
            pred = self.netmodel(input, grid)
        else:
            input = input.to(self.Device)
            pred = self.netmodel(input)

        pred = pred.reshape([pred.shape[0], self.grid_size, self.grid_size, -1])
        pred = self.out_norm.back(pred)

        return pred.detach().cpu().numpy()

    def predictor_value(self, input, input_para=None, parameterList=None, input_norm=False, setOpt=True):
        if not isinstance(parameterList, list):
            parameterList = [parameterList]

        pred_2d = self.predicter_2d(input, input_norm=input_norm)
        if input_para is None:
            # input_para = {
            #     "PressureStatic": 0,
            #     "TemperatureStatic": 1,
            #     "Density": 2,
            #     "PressureTotalW": 3,
            #     "TemperatureTotalW": 4,
            # }
            input_para = {
                "PressureStatic": 0,
                "TemperatureStatic": 1,
                "V2": 2,
                "W2": 3,
                "DensityFlow": 4,
            }

        grid = get_grid(real_path=os.path.join("..", "data"))
        post_pred = Post_2d(pred_2d, grid,
                            inputDict=input_para,
                            )

        Rst = []
        for parameter_Name in parameterList:
            if parameter_Name=="MassFlow":
                value = post_pred.get_MassFlow()
            else:
                value = getattr(post_pred, parameter_Name)
                value = post_pred.span_density_average(value[..., -1])

            if setOpt: #如果默认输出最优值
                Rst.append(value * self.MaxOrMIn(parameter_Name))
            else:
                Rst.append(value)

        return np.concatenate(Rst, axis=1)
    @staticmethod
    def MaxOrMIn(parameter):

        dict = {
        "Efficiency": -1, #越大越好
        "EfficiencyPoly": -1,
        "PressureRatioV": -1,
        "TemperatureRatioV": -1,
        "PressureLossR":  1,
        "EntropyStatic":  1,
        "MachIsentropic": 1,
        "Load": 1,
        "MassFlow": 1,
        }

        return dict[parameter]