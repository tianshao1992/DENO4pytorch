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

    def predicter_loader(self, input_data):
        """
        加载完整的模型预测输入的坐标
        Net_model 训练完成的模型
        input 模型的输入 shape:[num, input_dim]
        先转换数据，分批计算
        """
        # torch.utils.data.TensorDataset(input_data)
        loader = torch.utils.data.DataLoader(input_data,
                                             batch_size=32,
                                             shuffle=False,
                                             drop_last=False)
        pred = []
        for input in loader:
            if self.name in ("FNO", "UNet", "Transformer"):
                with torch.no_grad():
                    input = torch.tensor(np.tile(input[:, None, None, :], (1, 64, 64, 1)),dtype=torch.float32)
                    input = input.to(self.Device)
                    grid = feature_transform(input)
                    temp = self.netmodel(input, grid)
                    pred.append(temp.clone())
                    temp = None
            else:
                with torch.no_grad():
                    input = input.to(self.Device)
                    temp = self.netmodel(input)
                    pred.append(temp.clone())
                    temp = None

        pred = torch.cat(pred, dim=0)
        return pred

    def predictor_value(self, input,
                        input_para=None, parameterList=None,
                        input_norm=False, setOpt=True,
                        soft_constraint=None,
                        ):
        if parameterList is None:
            parameterList = [
                "PressureRatioV", "TemperatureRatioV",
                "PressureRatioW", "TemperatureRatioW",
                "Efficiency", "EfficiencyPoly",
                "PressureLossR", "EntropyStatic",
                "MachIsentropic", "Load", "MassFlow"]
        if not isinstance(parameterList, list):
            parameterList = [parameterList]

        if len(input)<32:
            pred_2d = self.predicter_2d(input, input_norm=input_norm)
            # pred_2d = pred_2d.cpu().numpy()
        else:
            pred_2d = self.predicter_loader(input)
            pred_2d = pred_2d.cpu().numpy()



        if input_para is None:
            input_para = {
                "PressureStatic": 0,
                "TemperatureStatic": 1,
                "V2": 2,
                "W2": 3,
                "DensityFlow": 4,
            }
        if soft_constraint is None:
            soft_constraint = []

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

            if setOpt and parameter_Name not in soft_constraint: #如果默认输出最优值 #如果是约束项就不用修改了
                value = value * self.MaxOrMIn(parameter_Name)

            if parameter_Name in soft_constraint:
                value = np.power((self.Constraint(parameter_Name) - value), 2)#将软约束作为一个目标r
                print(parameter_Name)
                print(value)

            Rst.append(value.copy())
            value = []

        return np.concatenate(Rst, axis=1)

    def predictor_hardConstraint(self, input, hardconstrList):
        pred = self.predictor_value(input,
                                    input_para=None, parameterList=hardconstrList,
                                    input_norm=False, setOpt=False,
                                    soft_constraint=None
                                    )
        Rst = []
        for ii, hardconstr in enumerate(hardconstrList):
            value = pred[:, ii:ii+1] - self.Constraint(hardconstr) # 实际值减去约束 ， 如果小于约束则<0 符合约束要求
            value = value * self.MaxOrMIn(hardconstr) #如果值是越大越好的，则乘以-1
            Rst.append(value)

            # print(hardconstr)
            # print(value)

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

    @staticmethod
    def Constraint(parameter):
        dict = {
        "Efficiency": 0, #越大越好
        "EfficiencyPoly": 0,
        "PressureRatioV": 2.0457,
        "TemperatureRatioV": 0,
        "PressureLossR":  0,
        "EntropyStatic":  0,
        "MachIsentropic": 0,
        "Load": 0,
        "MassFlow": 20.4
        }

        return dict[parameter]