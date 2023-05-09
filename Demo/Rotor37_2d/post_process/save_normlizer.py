# 获取确定的normlizer的pkl文件
import os
import numpy as np
import torch
from load_model import loaddata, rebuild_model
from utilizes_rotor37 import get_origin

if __name__ == "__main__":
    nameReal = "MLP"
    # train_number = 1250
    # for ii in range(10):
    #     _, _, x_normalizer_1, y_normalizer_1 = loaddata(nameReal, 1500, 150, shuffled=True)
    #     _, _, x_normalizer_2, y_normalizer_2 = loaddata(nameReal, 1000, 150, shuffled=True)
    #     print("================")
    #     print(y_normalizer_1.std - y_normalizer_2.std)
    #     print(y_normalizer_1.mean - y_normalizer_2.mean)
    #     print("================")
    # save_path_x = os.path.join("..", "data", "x_norm_" + str(train_number) + ".pkl")
    # save_path_y = os.path.join("..", "data", "y_norm_" + str(train_number) + ".pkl")
    # x_normalizer.save(save_path=save_path_x)
    # y_normalizer.save(save_path=save_path_y)

    #===========================================================================================#
    design, fields = get_origin(realpath=os.path.join("..", "data"), shuffled=True,
                                quanlityList=["Static Pressure", "Static Temperature"])  # 获取原始数据
    #
    for ii in range(20):
        num = (ii+1) * 100
        print(np.mean(torch.tensor(fields, dtype=torch.float64)[:num, :].numpy(), axis=(0, 1, 2,)))
        print(np.mean(fields[:num, :], axis=(0, 1, 2,)))
        print("================")


