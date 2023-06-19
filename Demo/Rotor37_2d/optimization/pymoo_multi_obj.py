import os
import numpy as np
import yaml
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo_optimizer import Rotor37Predictor, predictor_establish

def generate_tasks():
    yml_path = os.path.join("EPM_optmization_tasks.yml")
    task_1 = {
        'parameterList' :["Efficiency", "PressureRatioV"],
        'softconstrList' : [],
        'hardConstrList' : ["MassFlow"],
        'hardConstrIneqList' : [],
            }

    with open(yml_path, 'w') as f:
        yaml.dump(task_1, f)



if __name__ == "__main__":
    # 设置需要优化的函数
    name = 'FNO_1'
    input_dim = 28
    output_dim = 5
    work_load_path = os.path.join("..", "work_train_FNO2")
    # work_load_path = os.path.join("..", "work")

    model_all = predictor_establish(name, work_load_path)

    yml_path = os.path.join("EPM_optmization_tasks.yml")

    with open(yml_path) as f:
        config = yaml.full_load(f)

    dict = {}
    # 单个对象优化
    parameterList = [
        "Efficiency",
        "EfficiencyPoly",
        "PressureLossR",
    ]
    for ii, parameter in enumerate(parameterList):
        for task_id in range(8):
            config["task_" + str(task_id)]['parameterList'][0] = parameter
            problem = Rotor37Predictor(model_all, **config["task_" + str(task_id)])
            # 定义优化算法
            algorithm = NSGA2(pop_size=30)
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

            dict["task_" + str(task_id + ii*8) + "_sample"] = res.X
            dict["task_" + str(task_id + ii*8) + "_value"] = res.F

        # 保存数据
        np.savez(os.path.join("..", "data", "opt_data", 'EPM_optmization_tasks.npz'), **dict)
