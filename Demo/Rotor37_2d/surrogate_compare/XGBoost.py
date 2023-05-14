import xgboost as xgb
import numpy as np
import os

def train_model(train_x, train_y):
    # XGBoost训练过程
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160,
                             objective='reg:squarederror')
    # objective='reg:gamma'
    model.fit(train_x, train_y)

    return model

def get_pred(npz_path, n_train, parameter):

    data = np.load(npz_path)
    design = data["Design"]

    value = data[parameter].squeeze()

    train_x = design[:n_train]
    train_y = value[:n_train]

    valid_x = design[-400:]
    valid_y = value[-400:]

    model = train_model(train_x, train_y)

    pred = model.predict(valid_x)
    test_results = np.mean(np.abs((valid_y - pred) / valid_y))
    print(parameter + "_error = " + str(test_results))

    return  pred

if __name__ == "__main__":
    npz_path = os.path.join("data", "scalar_value.npz")
    data = np.load(npz_path)
    design = data["Design"]
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

    np.savez(os.path.join("data", "XGBoost.npz"), **dict_all)

