import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

import torch
import os
from post_process.model_predict import DLModelPost
from post_process.load_model import loaddata, rebuild_model, build_model_yml
from Utilizes.process_data import DataNormer, MatLoader, SquareMeshGenerator
from train_model.model_whole_life import WorkPrj
import matplotlib.pyplot as plt

# 定义目标函数
class SphereWithConstraint(Problem):

    def __init__(self):

        super().__init__(n_var=10, n_obj=2, n_ieq_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = np.sum((x - 0.1) ** 2, axis=1)
        f2 = np.sum((x - 0.5) ** 2, axis=1)

        out["F"] = np.column_stack([f1, f2])


class Rotor37Predictor(Problem):

    def __init__(self, model, parameterList=None):
        self.parameterList = parameterList
        self.model = model
        super().__init__(n_var=28, n_obj=len(parameterList), n_ieq_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.model.predictor_value(x, parameterList=self.parameterList)


if __name__ == "__main__":
    # 设置需要优化的函数
    name = 'FNO'
    input_dim = 28
    output_dim = 5
    work_load_path = os.path.join("..", "work")
    work_path = os.path.join(work_load_path, name)
    work = WorkPrj(work_path)

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    if os.path.exists(work.x_norm):
        norm_save_x = work.x_norm
        norm_save_y = work.y_norm
    else:
        norm_save_x = os.path.join("..", "data", "x_norm_1250.pkl")
        norm_save_y = os.path.join("..", "data", "y_norm_1250.pkl")

    x_normlizer = DataNormer([1, 1], method="mean-std", axis=0)
    x_normlizer.load(norm_save_x)
    y_normlizer = DataNormer([1, 1], method="mean-std", axis=0)
    y_normlizer.load(norm_save_y)

    if os.path.exists(work.yml):
        Net_model, inference, _, _ = build_model_yml(work.yml, Device, name=name)
        isExist = os.path.exists(work.pth)
        if isExist:
            checkpoint = torch.load(work.pth, map_location=Device)
            Net_model.load_state_dict(checkpoint['net_model'])
    else:
        Net_model, inference = rebuild_model(work_path, Device, name=name)
    model_all = DLModelPost(Net_model, Device,
                        name=name,
                        in_norm=x_normlizer,
                        out_norm=y_normlizer,
                        )
    parameterList = [
        "Efficiency",
        "EfficiencyPoly",
        # "PressureRatioV", "TemperatureRatioV",
        # "PressureLossR", "EntropyStatic",
        # "MachIsentropic", "Load",
    ]


    # 创建问题对象
    problem = Rotor37Predictor(model_all, parameterList=parameterList)
    # 定义优化算法
    algorithm = NSGA2(pop_size=10)
    # 进行优化
    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 100),
                   verbose=True)# 打印最优解

    print("最优解：", res.X)
    print("最优目标函数值：", res.F)

