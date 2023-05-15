# 获取所有样本8个参数的估计值
import numpy as np
import os
from Demo.Rotor37_2d.utilizes_rotor37 import get_grid, get_origin
from post_process.post_data import Post_2d


if __name__ == "__main__":
# 数据读入
    work_path = os.path.join("..", "data")
    isCreated = os.path.exists(work_path)
    if not isCreated: os.mkdir(work_path)

    grid = get_grid(real_path=os.path.join("..", "data"))
    design, field = get_origin(realpath=os.path.join("..", "data"),
                               quanlityList=["Static Pressure", "Static Temperature",
                                             'Absolute Total Temperature',  # 'Absolute Total Pressure',
                                             'Relative Total Temperature',  # 'Relative Total Pressure',
                                             "DensityFlow",
                                             # "Vxyz_X", "Vxyz_Y",
                                             ])
    true = field
    pred = true + np.random.rand(*true.shape) * 0.5 - 1
    input_para = {
        "PressureStatic": 0,
        "TemperatureStatic": 1,
        "TemperatureTotalV": 2,
        "TemperatureTotalW": 3,
        "DensityFlow": 4,
    }
    ii = 0
    post_true = Post_2d(true, grid,
                        inputDict=input_para,
                        )
    parameterList = [
        "PressureRatioV", "TemperatureRatioV",
        "Efficiency", "EfficiencyPoly",
        "PressureLossR", "EntropyStatic",
        "MachIsentropic", "Load"]

    all_dict = {}
    for parameter in parameterList:
        value_span = getattr(post_true, parameter)
        scalar = post_true.span_density_average(value_span[:, :, -1])
        all_dict.update({parameter: scalar})


    hub_out = 0.1948
    shroud_out = 0.2370
    MassFlow = post_true.span_space_average(post_true.DensityFlow[:, :, -1])*(shroud_out**2-hub_out**2)*np.pi
    all_dict.update({"MassFlow": MassFlow})
    all_dict.update({"Design": design})


    #保存数据
    np.savez(os.path.join(work_path, 'scalar_value.npz'), **all_dict)


