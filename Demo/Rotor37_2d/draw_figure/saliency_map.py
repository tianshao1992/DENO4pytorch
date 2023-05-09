# 绘制显著图
import numpy as np
import torch
import os
from post_process.model_predict import DLmodel
from post_process.load_model import loaddata, rebuild_model
from Utilizes.process_data import DataNormer, MatLoader, SquareMeshGenerator

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Function

class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_variables
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, (input > 0).type_as(input))
        return grad_input

def guided_relu_hook_function(module, grad_in, grad_out):
    # Replace all negative gradients with zero
    for idx in range(len(grad_in)):
        if grad_in[idx] is not None:
            grad_in[idx][grad_in[idx] < 0] = 0

def compute_saliency_maps(X, y, model):
    model.eval()
    X_var = Variable(X, requires_grad=True)
    y_var = Variable(torch.LongTensor([y]))
    saliency = None
    def hook_func(module, grad_in, grad_out):
        nonlocal saliency
        saliency = grad_in[0].data.numpy()
    hook_handle = model.conv2.register_backward_hook(hook_func)
    logits = model(X_var)
    loss = F.cross_entropy(logits, y_var)
    loss.backward()
    hook_handle.remove()
    saliency = saliency[0]
    saliency = saliency / saliency.max()
    return saliency




if __name__ == "__main__":
    # 加载模型
    name = 'FNO'
    input_dim = 28
    output_dim = 5
    work_load_path = os.path.join("..", "work")
    work_path = os.path.join(work_load_path, name)

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    x_normlizer = DataNormer([1, 1], method="mean-std", axis=0)
    norm_save_x = os.path.join("..", "data", "x_norm_1250.pkl")
    x_normlizer.load(norm_save_x)
    y_normlizer = DataNormer([1, 1], method="mean-std", axis=0)
    norm_save_y = os.path.join("..", "data", "y_norm_1250.pkl")
    y_normlizer.load(norm_save_y)

    Net_model, inference = rebuild_model(work_path, Device, name=name)
    model_all = DLmodel(Net_model, Device,
                        name=name,
                        in_norm=x_normlizer,
                        out_norm=y_normlizer,
                        )