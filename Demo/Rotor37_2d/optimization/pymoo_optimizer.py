import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
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

class Rotor37Predictor(Problem):

    def __init__(self, model, parameterList=None):
        self.parameterList = parameterList
        self.model = model
        super().__init__(n_var=28, n_obj=len(parameterList), n_ieq_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.model.predictor_value(x, parameterList=self.parameterList, setOpt=False)


def predictor_establish(name, work_load_path):
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
    return model_all

def calculate_hypervolume(front, reference_point):
    """
    计算超体积指标（Hypervolume Indicator）

    参数:
    - front: 二维数组，每一行表示一个前沿解的目标值
    - reference_point: 一维数组，参考点的目标值

    返回值:
    - hypervolume: 超体积指标的值
    """
    # 将参考点添加到前沿解中
    front = np.vstack((front, reference_point))

    # 对前沿解按照第一个目标值进行排序
    front = front[np.argsort(front[:, 0])]

    hypervolume = 0.0  # 初始化超体积指标的值
    last_volume = 1.0  # 上一层级的体积值

    # 逐个计算每一层级的体积值
    for i in range(front.shape[0]):
        current_volume = np.prod(front[i] - reference_point)  # 当前层级的体积值
        hypervolume += (last_volume - current_volume)  # 累加层级的体积值
        last_volume = current_volume  # 更新上一层级的体积值

    return hypervolume

# front = np.array([[4, 3], [2, 5], [1, 6], [3, 4]])
# # 参考点的目标值
# reference_point = np.array([0, 0])
# # 计算超体积指标
# hv = calculate_hypervolume(front, reference_point)
# print("Hypervolume:", hv)


if __name__ == "__main__":
    # 设置需要优化的函数
    name = 'FNO'
    input_dim = 28
    output_dim = 5
    work_load_path = os.path.join("..", "work")

    model_all = predictor_establish(name, work_load_path)

    parameterList = [
        "Efficiency",
        "EfficiencyPoly",
        "PressureRatioV",
        "TemperatureRatioV",
        "PressureRatioW",
        "TemperatureRatioW",
        "PressureLossR",
        "EntropyStatic",
        "MachIsentropic",
        "Load",
        "MassFlow"
    ]

    # 单个对象优化
    for parameter in parameterList:
        # 创建问题对象
        problem = Rotor37Predictor(model_all, parameterList=[parameter])
        # 定义优化算法
        # algorithm = NSGA2(pop_size=10)
        algorithm = GA(pop_size=20)
        # 进行优化
        res = minimize(problem,
                       algorithm,
                       termination=('n_gen', 500),
                       verbose=True,
                       save_history=True
                       )# 打印最优解

        # print("最优解：", res.X)
        # print("最优目标函数值：", res.F)

        # 保存到文件中
        dict = {}
        dict[parameter+"_sample"] = res.X
        dict[parameter + "_value"] = res.F

    # 保存数据
    np.savez(os.path.join("..", "data", "opt_data", 'sin_obj_minimize.npz'), **dict)





    # n_evals = np.array([e.evaluator.n_eval for e in res.history])
    # opt = np.array([e.opt[0].F for e in res.history])
    # plt.scatter(opt[:, 0], opt[:, 1])
    # plt.show()

