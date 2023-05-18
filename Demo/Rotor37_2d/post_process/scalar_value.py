# 获取所有样本8个参数的估计值
import numpy as np
import os
from Demo.Rotor37_2d.utilizes_rotor37 import get_grid, get_origin
from post_process.post_data import Post_2d


if __name__ == "__main__":
# 数据读入
    work_path = os.path.join("..", "data", "surrogate_data")
    isCreated = os.path.exists(work_path)
    if not isCreated: os.mkdir(work_path)

    grid = get_grid(real_path=os.path.join("..", "data"))
    design, field = get_origin(realpath=os.path.join("..", "data"), shuffled=True)
    true = field
    pred = true + np.random.rand(*true.shape) * 0.5 - 1
    ii = 0
    post_true = Post_2d(true, grid,
                        inputDict=None,
                        )
    parameterList = [
        "PressureRatioV", "TemperatureRatioV",
        "PressureRatioW", "TemperatureRatioW",
        "Efficiency", "EfficiencyPoly",
        "PressureLossR", "EntropyStatic",
        "MachIsentropic", "Load"]

    all_dict = {}
    for parameter in parameterList:
        value_span = getattr(post_true, parameter)
        scalar = post_true.span_density_average(value_span[:, :, -1])
        all_dict.update({parameter: scalar})

    MassFlow = post_true.get_MassFlow()
    all_dict.update({"MassFlow": MassFlow})
    all_dict.update({"Design": design})


    #保存数据
    np.savez(os.path.join(work_path, 'scalar_value.npz'), **all_dict)


