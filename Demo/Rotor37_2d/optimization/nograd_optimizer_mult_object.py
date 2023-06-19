import nevergrad as ng
import numpy as np
import torch
import os
from post_process.model_predict import DLModelPost
from post_process.load_model import loaddata, rebuild_model, build_model_yml
from Utilizes.process_data import DataNormer, MatLoader, SquareMeshGenerator
from train_model.model_whole_life import WorkPrj
import matplotlib.pyplot as plt

def fake_training(learning_rate: float, batch_size: int, architecture: str) -> float:
    # optimal for learning_rate=0.2, batch_size=4, architecture="conv"
    return (learning_rate - 0.2)**2 + (batch_size - 4)**2 + (0 if architecture == "conv" else 10)



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

    targetFunc = lambda x: model_all.predictor_value(x, parameterList=["PressureLossR"])
    # x_init = np.zeros([2, 28])
    # Rst = targetFunc(x_init)
    # print(Rst)

    # 设置各个变量的范围
    para = ng.p.Array(shape=(1, 28), lower=0, upper=1)

    #选择并设置优化器
    optimizer = ng.optimizers.NSGA2(parametrization=para, budget=200, num_workers=10)
    # optimizer = ng.optimizers.LhsDE(parametrization=para, budget=200)

    #获得优化结果
    recommendation = optimizer.minimize(targetFunc)
    print(recommendation.value)

    # for i in range(10):
    #     candidate = optimizer.ask()
    #     value = targetFunc(candidate.value)
    #     optimizer.tell(candidate, value)
    #     optimizer.provide_recommendation()
    #     print(optimizer.current_bests["minimum"])
    #     # optimizer.watch(watch = "x", note = "iteration " + str(i))
    # loss_values = []
    # for _ in range(200):
    #     # 提取当前参数
    #     x = optimizer.ask()
    #     # 计算损失函数值
    #     loss = targetFunc(x.value)
    #     # 将损失函数值添加到列表中
    #     loss_values.append(loss)
    #     # 提供损失函数值给优化器
    #     optimizer.tell(x, loss)
    #
    # # 绘制优化收敛曲线
    # plt.plot(np.arange(len(loss_values)), np.concatenate(loss_values, axis=0))
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.show()