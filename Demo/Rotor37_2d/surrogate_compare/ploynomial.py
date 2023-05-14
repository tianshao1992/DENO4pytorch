import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# learningRate学习率，Loopnum迭代次数

def get_pred(npz_path, n_train, parameter):

    data = np.load(npz_path)
    design = data["Design"]

    value = data[parameter].squeeze()

    train_x = design[:n_train]
    train_y = value[:n_train]

    valid_x = design[-400:]
    valid_y = value[-400:]

    # 定义多项式特征转换器
    degree = 3  # 多项式的次数
    poly_features = PolynomialFeatures(degree=degree)

    # 进行多项式特征转换
    X_poly = poly_features.fit_transform(train_x)
    model = LinearRegression()
    model.fit(X_poly, train_y)

    X_poly_valid = poly_features.transform(valid_x)
    pred = model.predict(X_poly_valid)

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

    n_trainList = [500, 1000, 1500, 2000, 2500]


    for parameter in parameterList:
        data_box = np.zeros([400, 5])
        for ii,  n_train in enumerate(n_trainList):
            pred = get_pred(npz_path, n_train, parameter)
            data_box[:, ii] = pred.copy()
        dict_all.update({parameter: data_box})

    np.savez(os.path.join("data", "ploynomial.npz"), **dict_all)



