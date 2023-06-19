import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
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

    def __init__(self, model,  # 软约束包含于parameterList, 硬约束不包含于parameterList
                 parameterList=None,
                 softconstrList=None,
                 hardConstrList=None,
                 hardConstrIneqList=None,
                ):

        self.model = model
        self.parameterList = parameterList
        self.softconstrList = softconstrList
        self.hardConstrList = hardConstrList
        self.hardConstrIneqList = hardConstrIneqList

        super().__init__(n_var=28,
                         n_obj=len(parameterList),
                         n_ieq_constr=len(hardConstrIneqList), n_eq_constr=len(hardConstrList),
                         xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.model.predictor_value(x, parameterList=self.parameterList, setOpt=True) # 注意 这里修改过了。
        # 约束设置
        if self.hardConstrList is not None:
            if  len(self.hardConstrList) != 0:
                out["H"] = self.model.predictor_hardConstraint(x, hardconstrList=self.hardConstrList)
        if self.hardConstrIneqList is not None:
            if  len(self.hardConstrIneqList) != 0:
                out["G"] = self.model.predictor_hardConstraint(x, hardconstrList=self.hardConstrIneqList)


def predictor_establish(name, work_load_path):

    nameReal = name.split("_")[0]
    id = None
    if len(name.split("_")) == 2:
        id = int(name.split("_")[1])

    work_path = os.path.join(work_load_path, name)
    work = WorkPrj(work_path)
    # if torch.cuda.is_available():
    #     Device = torch.device('cuda')
    # else:
    Device = torch.device('cpu') #优化就在CPU

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
        Net_model, inference, _, _ = build_model_yml(work.yml, Device, name=nameReal)
        isExist = os.path.exists(work.pth)
        if isExist:
            checkpoint = torch.load(work.pth, map_location=Device)
            Net_model.load_state_dict(checkpoint['net_model'])
    else:
        Net_model, inference = rebuild_model(work_path, Device, name=nameReal)
    model_all = DLModelPost(Net_model, Device,
                        name=nameReal,
                        in_norm=x_normlizer,
                        out_norm=y_normlizer,
                        )
    return model_all


if __name__ == "__main__":
    # 设置需要优化的函数
    name = 'FNO'
    input_dim = 28
    output_dim = 5
    # work_load_path = os.path.join("..", "work_train_FNO2")
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
    dict = {}
    # for parameter in parameterList:
        # 创建问题对象
    problem = Rotor37Predictor(model_all,
                               parameterList=["Efficiency", "PressureRatioV"],
                               softconstrList=[],
                               hardConstrList=["MassFlow"],
                               hardConstrIneqList=[],
                              )
    # 定义优化算法
    algorithm = NSGA2(pop_size=10)
    # algorithm = GA(pop_size=20)
    # 进行优化
    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 200),
                   verbose=True,
                   save_history=True
                   )# 打印最优解

    print("最优解：", res.X)
    print("最优目标函数值：", res.F)

    # 保存到文件中

    # dict[parameter+"_sample"] = res.X
    # dict[parameter + "_value"] = res.F

        # np.savez(os.path.join("..", "data", "opt_data", 'sin_obj_minimize.npz'), **dict)

    # 保存数据
    # np.savez(os.path.join("..", "data", "opt_data", 'sin_obj_maximize.npz'), **dict)





    # n_evals = np.array([e.evaluator.n_eval for e in res.history])
    # opt = np.array([e.opt[0].F for e in res.history])
    # plt.scatter(opt[:, 0], opt[:, 1])
    # plt.show()

