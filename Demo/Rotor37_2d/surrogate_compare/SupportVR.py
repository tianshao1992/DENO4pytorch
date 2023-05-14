from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def get_pred(npz_path, n_train, parameter):
    data = np.load(npz_path)
    design = data["Design"]

    value = data[parameter].squeeze()

    train_x = design[:n_train]
    train_y = value[:n_train]

    valid_x = design[-400:]
    valid_y = value[-400:]

    svm_regressor = SVR(kernel='linear')  # 创建支持向量机回归模型
    svm_regressor.fit(train_x, train_y)  # 拟合训练数据

    pred = svm_regressor.predict(valid_x)
    test_results = np.mean(np.abs((valid_y - pred) / valid_y))
    print(parameter + "_error = " + str(test_results))

    return pred


if __name__ == "__main__":
    npz_path = os.path.join("data", "scalar_value.npz")
    dict_all = {}

    parameterList = [
    "PressureRatioV", "TemperatureRatioV",
    "Efficiency", "EfficiencyPoly",
    "PressureLossR", "EntropyStatic",
    "MachIsentropic", "Load",
    "MassFlow"]
    for parameter in parameterList:

        n_trainList = [500, 1000, 1500, 2000, 2500]

        for parameter in parameterList:
            data_box = np.zeros([400, 5])
            for ii, n_train in enumerate(n_trainList):
                pred = get_pred(npz_path, n_train, parameter)
                data_box[:, ii] = pred.copy()
            dict_all.update({parameter: data_box})

        np.savez(os.path.join("data", "SVR.npz"), **dict_all)