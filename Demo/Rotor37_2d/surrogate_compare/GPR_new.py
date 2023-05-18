import torch
import os
import numpy as np
import gpytorch
from post_process.load_model import get_noise

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def get_pred(npz_path, n_train, n_noise, parameter):
    data = np.load(npz_path)
    design = torch.tensor(data["Design"], dtype=torch.double)
    value = torch.tensor(data[parameter].squeeze(), dtype=torch.double)

    train_x = design[:n_train]
    train_y = value[:n_train]
    noise = get_noise(train_y.numpy().shape, n_noise) * np.mean(train_y.numpy())
    train_y = train_y + torch.tensor(noise, dtype=torch.double)

    valid_x = design[-400:]
    valid_y = value[-400:]

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    training_iter = 100
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -1 * mll(output, train_y)
        # loss = sum(loss)
        loss.backward()
        if i % 10 == 9:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        predictions = likelihood(model(valid_x))
        mean = predictions.mean
        # lower, upper = predictions.confidence_region()

    test_results = np.mean(np.abs((valid_y - mean) / valid_y).numpy(), axis=0)
    print(parameter + "_error = " + str(test_results))

    return mean.numpy()


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

        n_trainList = [800, 1000, 1500, 2000, 2500]
        n_noiseList = [0, 0.005, 0.01, 0.05, 0.1]

        data_box_num = np.zeros([400, 5])
        for ii, n_train in enumerate(n_trainList):
            pred = get_pred(npz_path, n_train, 0, parameter)
            data_box_num[:, ii] = pred.copy()
        dict_num.update({parameter: data_box_num})
        pred = None

        data_box_noise = np.zeros([400, 5])
        for ii, n_noise in enumerate(n_noiseList):
            pred = get_pred(npz_path, 2500, n_noise, parameter)
            data_box_noise[:, ii] = pred.copy()
        dict_noise.update({parameter: data_box_noise})
        pred = None

        np.savez(os.path.join("..", "data", "surrogate_data", "GPR_num.npz"), **dict_num)
        np.savez(os.path.join("..", "data", "surrogate_data", "GPR_noise.npz"), **dict_noise)

    # np.savez(os.path.join("data", "GPR.npz"), **dict_all)



