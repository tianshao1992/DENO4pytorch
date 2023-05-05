# 获取确定的normlizer的pkl文件
import os
from load_model import loaddata, rebuild_model

if __name__ == "__main__":
    nameReal = "MLP"
    train_number = 1250
    train_loader, valid_loader, x_normalizer, y_normalizer = loaddata(nameReal, train_number, 150)
    save_path_x = os.path.join("..", "data", "x_norm_" + str(train_number) + ".pkl")
    save_path_y = os.path.join("..", "data", "y_norm_" + str(train_number) + ".pkl")
    x_normalizer.save(save_path=save_path_x)
    y_normalizer.save(save_path=save_path_y)