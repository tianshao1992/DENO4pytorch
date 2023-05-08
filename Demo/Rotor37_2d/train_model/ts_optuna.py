import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable  # 获取变量
import time
import optuna


def train(batch_size, learning_rate, lossfunc, opt, hidden_layer, activefunc, weightdk, momentum):  # 选出一些超参数
    trainset_num = 800
    testset_num = 50

    train_dataset = myDataset(trainset_num)
    test_dataset = myDataset(testset_num)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 创建CNN模型， 并设置损失函数及优化器
    model = MLP(hidden_layer, activefunc).cuda()
    # print(model)
    if lossfunc == 'MSE':
        criterion = nn.MSELoss().cuda()
    elif lossfunc == 'MAE':
        criterion = nn.L1Loss()

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weightdk)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weightdk, momentum=momentum)
    # 训练过程
    for epoch in range(num_epoches):
        # 训练模式
        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels, _ = data
            inputs = Variable(inputs).float().cuda()
            labels = Variable(labels).float().cuda()
            # 前向传播
            out = model(inputs)
            # 可以考虑加正则项
            train_loss = criterion(out, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

    model.eval()

    testloss = test()  # 返回测试集合上的MAE
    print('Test MAE = ', resloss)
    return resloss


def objective(trail):
    batchsize = trail.suggest_int('batchsize', 1, 16)
    lr = trail.suggest_float('lr', 1e-4, 1e-2, step=0.0001)
    lossfunc = trail.suggest_categorical('loss', ['MSE', 'MAE'])
    opt = trail.suggest_categorical('opt', ['Adam', 'SGD'])
    hidden_layer = trail.suggest_int('hiddenlayer', 20, 1200)
    activefunc = trail.suggest_categorical('active', ['relu', 'sigmoid', 'tanh'])
    weightdekey = trail.suggest_float('weight_dekay', 0, 1, step=0.01)
    momentum = trail.suggest_float('momentum', 0, 1, step=0.01)
    loss = train(batchsize, lr, lossfunc, opt, hidden_layer, activefunc, weightdekey, momentum)
    return loss


if __name__ == '__main__':
    st = time.time()
    study = optuna.create_study(study_name='test', direction='minimize')
    study.optimize(objective, n_trials=500)
    print(study.best_params)
    print(study.best_trial)
    print(study.best_trial.value)
    print(time.time() - st)
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_slice(study).show()

