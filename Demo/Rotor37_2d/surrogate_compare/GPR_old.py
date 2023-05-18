import torch
import os
import numpy as np
import gpytorch

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def get_pred(npz_path, n_train, parameter):
    Device = torch.device('cpu')
    data = np.load(npz_path)
    design = torch.tensor(data["Design"], dtype=torch.double).to(Device)
    value = torch.tensor(data[parameter].squeeze(), dtype=torch.double).to(Device)

    train_x = design[:n_train]
    train_y = value[:n_train]

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
        loss = -mll(output, train_y)
        # loss = sum(loss)
        loss.backward()
        if i % 5 == 4:
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

    np.savez(os.path.join("data", "GPR.npz"), **dict_all)



