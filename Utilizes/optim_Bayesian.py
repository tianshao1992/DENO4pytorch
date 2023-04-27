#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/2/19 23:00
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : optim_Bayesian.py
"""
import torch
import torch.nn as nn
from enum import Enum

from numpy import pi
from . import util


# Docstring:
# https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard


class Sampler(Enum):
    HMC = 1
    RMHMC = 2
    HMC_NUTS = 3
    # IMPORTANCE = 3
    # MH = 4


class Integrator(Enum):
    EXPLICIT = 1
    IMPLICIT = 2
    S3 = 3
    SPLITTING = 4
    SPLITTING_RAND = 5
    SPLITTING_KMID = 6


class Metric(Enum):
    HESSIAN = 1
    SOFTABS = 2
    JACOBIAN_DIAG = 3


def collect_gradients(log_prob, params):
    """
    求导
    """

    if isinstance(log_prob, tuple):
        log_prob[0].backward()
        params_list = list(log_prob[1])
        params = torch.cat([p.flatten() for p in params_list])
        params.grad = torch.cat([p.grad.flatten() for p in params_list])
    else:
        params.grad = torch.autograd.grad(log_prob, params)[0]
    return params


def gibbs(params):
    """
    动量采样
    """
    dist = torch.distributions.Normal(torch.zeros_like(params), torch.ones_like(params))
    return dist.sample()


def leapfrog(params, momentum, log_prob_func, steps=10, step_size=0.1, normalizing_const=1., inv_mass=None,
             sampler=Sampler.HMC, integrator=Integrator.IMPLICIT):
    """
    leapfrog算法实现
    """

    params = params.clone()
    momentum = momentum.clone()
    # TodO detach graph when storing ret_params for memory saving
    if sampler == Sampler.HMC and integrator != Integrator.SPLITTING and integrator != Integrator.SPLITTING_RAND and integrator != Integrator.SPLITTING_KMID:
        def params_grad(p):
            p = p.detach().requires_grad_()
            log_prob = log_prob_func(p)  # 计算动能
            # log_prob.backward()
            p = collect_gradients(log_prob, p)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return p.grad

        ret_params = []
        ret_momenta = []
        momentum += 0.5 * step_size * params_grad(params)
        for n in range(steps):
            if inv_mass is None:
                params = params + step_size * momentum  # /normalizing_const
            else:
                # Assum G is diag here so 1/Mass = G inverse
                if type(inv_mass) is list:
                    i = 0
                    for block in inv_mass:
                        it = block[0].shape[0]
                        params[i:it + i] = params[i:it + i] + step_size * torch.matmul(
                            block, momentum[i:it + i].view(-1, 1)).view(-1)  # /normalizing_const
                        i += it
                elif len(inv_mass.shape) == 2:
                    params = params + step_size * \
                             torch.matmul(inv_mass, momentum.view(-1, 1)
                                          ).view(-1)  # /normalizing_const
                else:
                    params = params + step_size * inv_mass * momentum  # /normalizing_const
            p_grad = params_grad(params)
            momentum += step_size * p_grad
            ret_params.append(params.clone())
            ret_momenta.append(momentum.clone())
        # only need last for Hamiltoninian check (see p.14) https://arxiv.org/pdf/1206.1901.pdf
        ret_momenta[-1] = ret_momenta[-1] - 0.5 * step_size * p_grad.clone()
        return ret_params, ret_momenta
    else:
        raise NotImplementedError()


def acceptance(h_old, h_new):
    """
    用于计算是否接受当前采样参数
    """
    return float(-h_new + h_old)


# Adaptation p.15 No-U-Turn samplers Algo 5


def hamiltonian(params, momentum, log_prob_func, sampler=Sampler.HMC):
    """
    计算总能量：动能+势能
    """
    # Hamiltonian MCMC(哈密顿系统+MCMC)
    if sampler == Sampler.HMC:
        if type(log_prob_func) is not list:
            log_prob = log_prob_func(params)

            if util.has_nan_or_inf(log_prob):
                print('Invalid log_prob: {}, params: {}'.format(log_prob, params))
                raise util.LogProbError()

        potential = -log_prob  # /normalizing_const   #势能计算
        kinetic = 0.5 * torch.dot(momentum, momentum)  # /normalizing_const  #动能计算
        hamiltonian = potential + kinetic
        # hamiltonian = hamiltonian
    else:
        raise NotImplementedError()
    return hamiltonian


def sample(log_prob_func, params_init, num_samples=10, num_steps_per_sample=10, step_size=0.1, burn=0, inv_mass=None,
           normalizing_const=1., sampler=Sampler.HMC, integrator=Integrator.IMPLICIT, metric=Metric.HESSIAN,
           debug=False, store_on_GPU=True):
    """
    HMC采样
    """

    # Needed for memory moving i.e. move samples to CPU RAM so lookup GPU device
    device = params_init.device

    if params_init.dim() != 1:
        raise RuntimeError('params_init must be a 1d tensor.')

    if burn >= num_samples:
        raise RuntimeError('burn must be less than num_samples.')

    NUTS = False

    # Invert mass matrix once (As mass is used in Gibbs resampling step)
    mass = None
    if inv_mass is not None:
        if type(inv_mass) is list:
            mass = []
            for block in inv_mass:
                mass.append(torch.inverse(block))
        # Assum G is diag here so 1/Mass = G inverse
        elif len(inv_mass.shape) == 2:
            mass = torch.inverse(inv_mass)
        elif len(inv_mass.shape) == 1:
            mass = 1 / inv_mass

    params = params_init.clone().requires_grad_()
    if not store_on_GPU:
        ret_params = [params.clone().detach().cpu()]
    else:
        ret_params = [params.clone()]

    num_rejected = 0
    # if sampler == Sampler.HMC:
    util.progress_bar_init('Sampling ({}; {})'.format(
        sampler, integrator), num_samples, 'Samples')
    for n in range(num_samples):  # 设置的采样总数
        util.progress_bar_update(n)
        try:  # 采样动量,从(num=模型参数)个标准正态分布中采样动量
            momentum = gibbs(params)
            # 计算总的能量 H= 动能+势能
            ham = hamiltonian(params, momentum, log_prob_func, sampler=sampler)
            # 蛙跳法
            leapfrog_params, leapfrog_momenta = leapfrog(params, momentum, log_prob_func, sampler=sampler,
                                                         integrator=integrator, steps=num_steps_per_sample,
                                                         step_size=step_size, inv_mass=inv_mass,
                                                         normalizing_const=normalizing_const)

            params = leapfrog_params[-1].to(device).detach().requires_grad_()
            momentum = leapfrog_momenta[-1].to(device)
            new_ham = hamiltonian(params, momentum, log_prob_func, sampler=sampler)
            # new_ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, normalizing_const=normalizing_const, sampler=sampler, integrator=integrator, metric=metric)
            rho = min(0., acceptance(ham, new_ham))

            if rho >= torch.log(torch.rand(1)):
                if debug == 1:
                    print('Accept rho: {}'.format(rho))
                if n > burn:
                    if store_on_GPU:
                        ret_params.append(leapfrog_params[-1])
                    else:
                        # Store samples on CPU
                        ret_params.append(leapfrog_params[-1].cpu())
                        # ret_params.extend([lp.detach().cpu() for lp in leapfrog_params])
            else:
                num_rejected += 1
                params = ret_params[-1].to(device)
                if n > burn:
                    # leapfrog_params = ret_params[-num_steps_per_sample:] ### Might want to remove grad as wastes memory
                    if store_on_GPU:
                        ret_params.append(ret_params[-1].to(device))
                    else:
                        # Store samples on CPU
                        ret_params.append(ret_params[-1].cpu())
                if debug == 1:
                    print('REJECT')

        except util.LogProbError:
            num_rejected += 1
            params = ret_params[-1].to(device)
            if n > burn:
                # leapfrog_params = ret_params[-num_steps_per_sample:] ### Might want to remove grad as wastes memory
                if store_on_GPU:
                    ret_params.append(ret_params[-1].to(device))
                else:
                    # Store samples on CPU
                    ret_params.append(ret_params[-1].cpu())
            if debug == 1:
                print('REJECT')
            if NUTS and n == burn:
                step_size = eps_bar
                print('Final Adapted Step Size: ', step_size)

        if not store_on_GPU:  # i.e. delete stuff left on GPU
            # This adds approximately 50% to runtime when using colab 'Tesla P100-PCIE-16GB'
            # but leaves no memory footprint on GPU after use in normal HMC mode. (not split)
            # Might need to check if variables exist as a log prob error could occur before they are assigned!
            momentum = None
            leapfrog_params = None
            leapfrog_momenta = None
            ham = None
            new_ham = None

            del momentum, leapfrog_params, leapfrog_momenta, ham, new_ham
            torch.cuda.empty_cache()

            # var_names = ['momentum', 'leapfrog_params', 'leapfrog_momenta', 'ham', 'new_ham']
            # [util.gpu_check_delete(var, locals()) for var in var_names]
            # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    util.progress_bar_end('Acceptance Rate {:.2f}'.format(
        1 - num_rejected / num_samples))  # need to adapt for burn
    if NUTS and debug == 2:
        return list(map(lambda t: t.detach(), ret_params)), step_size
    elif debug == 2:
        return list(map(lambda t: t.detach(), ret_params)), 1 - num_rejected / num_samples
    else:
        return list(map(lambda t: t.detach(), ret_params))


def define_model_log_prob(model, model_loss, x, y, params_flattened_list, params_shape_list, tau_list, tau_out,
                          normalizing_const=1., predict=False, prior_scale=1.0, device='cpu'):
    """
    定义计算势能的函数
    """

    fmodel = util.make_functional(model)
    dist_list = []
    for tau in tau_list:
        # 按照网络w,b的数量建立高斯分布
        dist_list.append(torch.distributions.Normal(torch.zeros_like(tau_list[0]), tau ** -0.5))  # 生成均值为0，方差为1的正态分布

    def log_prob_func(params):
        """
        计算势能
        """
        params_unflattened = util.unflatten(model, params)  # reshape为w,b尺寸

        i_prev = 0
        # Set l2_reg to be on the same device as params
        l_prior = torch.zeros_like(params[0], requires_grad=True)
        for weights, index, shape, dist in zip(model.parameters(), params_flattened_list, params_shape_list, dist_list):
            # weights.data = params[i_prev:index+i_prev].reshape(shape)
            w = params[i_prev:index + i_prev]
            l_prior = dist.log_prob(w).sum() + l_prior  # 计算输入数据w在分布中的概率密度对数
            i_prev += index

        # Sample prior if no data
        if x is None:
            # print('hi')
            return l_prior / prior_scale

        x_device = x.to(device)
        y_device = y.to(device)

        output = fmodel(x_device, params=params_unflattened)

        if model_loss is 'binary_class_linear_output':
            crit = nn.BCEWithLogitsLoss(reduction='sum')
            ll = - tau_out * (crit(output, y_device))
        elif model_loss is 'multi_class_linear_output':
            #         crit = nn.MSELoss(reduction='mean')
            crit = nn.CrossEntropyLoss(reduction='sum')
            #         crit = nn.BCEWithLogitsLoss(reduction='sum')
            ll = - tau_out * (crit(output, y_device.long().view(-1)))
            # ll = - tau_out *(torch.nn.functional.nll_loss(output, y.long().view(-1)))
        elif model_loss is 'multi_class_log_softmax_output':
            ll = - tau_out * \
                 (torch.nn.functional.nll_loss(output, y_device.long().view(-1)))

        elif model_loss is 'regression':
            # crit = nn.MSELoss(reduction='sum')
            ll = - 0.5 * tau_out * ((output - y_device) ** 2).sum(0)  # sum(0)

        elif callable(model_loss):
            # Assume defined custom log-likelihood.
            ll = - model_loss(output, y_device).sum(0)
        else:
            raise NotImplementedError()

        if torch.cuda.is_available():
            del x_device, y_device
            torch.cuda.empty_cache()

        if predict:
            return (ll + l_prior / prior_scale), output
        else:
            return (ll + l_prior / prior_scale)

    return log_prob_func


def sample_model(model, x, y, params_init, model_loss, num_samples=10, num_steps_per_sample=10, step_size=0.1, burn=0,
                 inv_mass=None, normalizing_const=1., sampler=Sampler.HMC, integrator=Integrator.IMPLICIT, debug=False,
                 tau_out=1., tau_list=None, store_on_GPU=True):
    """
    从NN模型中对权重进行采样
    参数
    ----------
    model : 神经网络-
    x : 用来定义log概率的输入训练数据
    y : 用来定义log概率的输出训练数据
    params_init : 神经网络的初始参数
    model_loss : {'binary_class_linear_output', 'multi_class_linear_output', 'multi_class_log_softmax_output', 'regression'} or function
        This determines the likelihood to be used for the model. The options correspond to:
        * 'binary_class_linear_output': model has linear output and using binary cross entropy,
        * 'multi_class_linear_output': model has linear output and using cross entropy,
        * 'multi_class_log_softmax_output': model has log softmax output and using cross entropy,
        * 'regression': model has linear output and using Gaussian likelihood,
        * function: function of the form func(y_pred, y_true). It should return a vector (N,), where N is the number of data points.
    num_samples : 采样个数
    num_steps_per_sample : 常记为L
    step_size : 数值计算时的步长
    burn : 定义的燃烧期
    inv_mass : mass矩阵的逆矩阵
    normalizing_const : 常设置为1
    sampler : 设置HMC的采样类型{Sampler.HMC, Sampler.RMHMC, Sampler.HMC_NUTS}
    integrator : 设置leapfrog的整合方式:{Integrator.EXPLICIT, Integrator.IMPLICIT, Integrator.SPLITTING,Integrator.SPLITTING_RAND, Integrator.SPLITTING_KMID}.
    debug : 调试模式{0, 1, 2}
    tau_out : 似然输出精度
    tau_list : 每层参数的每一组对应的先验的张量，此处假设为高斯先验。
    """

    device = params_init.device  # cpu or gpu
    params_shape_list = []  # 网络参数形状列表
    params_flattened_list = []  # 网络每层参数个数列表
    build_tau = False
    if tau_list is None:
        tau_list = []
        build_tau = True
    for weights in model.parameters():
        params_shape_list.append(weights.shape)
        params_flattened_list.append(weights.nelement())
        if build_tau:
            tau_list.append(torch.tensor(1.))

    # 定义计算势能的函数
    log_prob_func = define_model_log_prob(model, model_loss, x, y, params_flattened_list, params_shape_list, tau_list,
                                          tau_out, normalizing_const=normalizing_const, device=device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清空显存缓冲区

    return sample(log_prob_func, params_init, num_samples=num_samples,
                  num_steps_per_sample=num_steps_per_sample, step_size=step_size, burn=burn, inv_mass=inv_mass,
                  normalizing_const=normalizing_const, sampler=sampler, integrator=integrator,
                  store_on_GPU=store_on_GPU)
