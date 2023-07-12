import xgboost as xgb
import numpy as np
import os
from post_process.load_model import get_noise

def train_model(train_x, train_y):
    # XGBoost训练过程
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160,
                             objective='reg:squarederror')
    # objective='reg:gamma'
    model.fit(train_x, train_y)

    return model

def get_pred(npz_path, n_train, n_noise, parameter):

    data = np.load(npz_path)
    design = data["Design"]

    value = data[parameter].squeeze()

    train_x = design[:n_train]
    train_y = value[:n_train]
    noise = get_noise(train_y.shape, n_noise) * np.std(train_y) * 2
    train_y = train_y + noise

    valid_x = design[-400:]
    valid_y = value[-400:]

    model = train_model(train_x, train_y)

    pred = model.predict(valid_x)
    test_results = np.mean(np.abs((valid_y - pred) / valid_y))
    print(parameter + "_error = " + str(test_results))

    return  pred

if __name__ == "__main__":
    npz_path = os.path.join("..", "data", "surrogate_data", "scalar_value.npz")
    dict_num = {}
    dict_noise = {}

    parameterList = [
        "PressureRatioV", "TemperatureRatioV",
        "Efficiency", "EfficiencyPoly",
        "PressureLossR", "EntropyStatic",
        "MachIsentropic", "Load",
        "MassFlow"]
    for parameter in parameterList:

        n_trainList = [500, 1000, 1500, 2000, 2500]
        n_noiseList = [0, 0.005, 0.01, 0.05, 0.1]

        data_box_num = np.zeros([400, 5])
        for ii, n_train in enumerate(n_trainList):
            pred = get_pred(npz_path, n_train, 0, parameter)
            data_box_num[:, ii] = pred.copy()
        dict_num.update({parameter: data_box_num})

        data_box_noise = np.zeros([400, 5])
        for ii, n_noise in enumerate(n_noiseList):
            pred = get_pred(npz_path, 2500, n_noise, parameter)
            data_box_noise[:, ii] = pred.copy()
        dict_noise.update({parameter: data_box_noise})

    np.savez(os.path.join("..", "data", "surrogate_data", "XGB_num.npz"), **dict_num)
    np.savez(os.path.join("..", "data", "surrogate_data", "XGB_noise.npz"), **dict_noise)

