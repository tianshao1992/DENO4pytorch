#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/2/19 23:24
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : Bayesian_util.py
"""
import torch
import torch.nn as nn
import Utilizes.util as util
import Utilizes.optim_Bayesian as Bayesian
import numpy as np
import time


def build_lists(models, n_params_single=None, tau_priors=None, tau_likes=0.1, pde=False):
    """_summary_
    构建关于网络参数(尺寸，个数)等列表
    """

    if n_params_single is not None:
        n_params = [n_params_single]
    else:
        n_params = []

    if isinstance(tau_priors, list) or tau_priors is None:
        build_tau_priors = False
    else:
        build_tau_priors = True
        tau_priors_elt = tau_priors
        tau_priors = []

    if isinstance(tau_likes, list):
        build_tau_likes = False
    else:
        build_tau_likes = True
        tau_likes_elt = tau_likes
        tau_likes = []

    params_shape_list = []
    params_flattened_list = []

    if build_tau_priors and n_params_single is not None:
        for _ in range(n_params_single):
            params_flattened_list.append(1)
            params_shape_list.append(1)
            tau_priors.append(tau_priors_elt)

    for model in models:
        n_params.append(util.flatten(model).shape[0])
        if build_tau_likes:
            tau_likes.append(tau_likes_elt)
        for weights in model.parameters():
            params_shape_list.append(weights.shape)
            params_flattened_list.append(weights.nelement())
            if build_tau_priors:
                tau_priors.append(tau_priors_elt)

    # if we deal with pde then we also have data of residual
    if pde and build_tau_likes:
        tau_likes.append(tau_likes_elt)

    n_params = list(np.cumsum(n_params))

    return params_shape_list, params_flattened_list, n_params, tau_priors, tau_likes


def define_model_log_prob_bpinns(models, model_loss, data, tau_priors=None, tau_likes=None, predict=False,
                                 prior_scale=1.0, n_params_single=None, pde=False):
    """
    返回log_prob_func函数
    """

    _, params_flattened_list, n_params, tau_priors, tau_likes = build_lists(models, n_params_single, tau_priors,
                                                                            tau_likes, pde)

    fmodel = []
    for model in models:
        fmodel.append(util.make_functional(model))

    if tau_priors is not None:
        dist_list = []
        for tau in tau_priors:
            dist_list.append(torch.distributions.Normal(0, tau ** -0.5))

    def log_prob_func(params):

        params_unflattened = []
        if n_params_single is not None:
            params_single = params[:n_params[0]]
            for i in range(len(models)):
                params_unflattened.append(util.unflatten(models[i], params[n_params[i]:n_params[i + 1]]))
        else:
            params_single = None
            for i in range(len(models)):
                if i == 0:
                    params_unflattened.append(util.unflatten(models[i], params[:n_params[i]]))
                else:
                    params_unflattened.append(util.unflatten(models[i], params[n_params[i - 1]:n_params[i]]))

        l_prior = torch.zeros_like(params[0], requires_grad=True)
        if tau_priors is not None:
            i_prev = 0
            for index, dist in zip(params_flattened_list, dist_list):
                w = params[i_prev:index + i_prev]
                l_prior = dist.log_prob(w).sum() + l_prior
                i_prev += index

        def gradients(outputs, inputs):
            return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

        ll, output = model_loss(data, fmodel, params_unflattened, tau_likes, gradients, params_single)

        if predict:
            return (ll + l_prior / prior_scale), output
        else:
            return (ll + l_prior / prior_scale)

    return log_prob_func


def sample_model_bpinns(models, data, model_loss, num_samples, num_steps_per_sample, step_size, burn, pde, pinns,
                        epochs, device, inv_mass=None, normalizing_const=1., sampler=Bayesian.Sampler.HMC,
                        integrator=Bayesian.Integrator.IMPLICIT, metric=Bayesian.Metric.HESSIAN, debug=False,
                        tau_priors=None, tau_likes=None, store_on_GPU=True, n_params_single=None, params_init_val=None):
    """
    HMC采样
    """
    if n_params_single is not None:
        params_init = torch.zeros([n_params_single]).to(device)
    else:
        params_init = torch.Tensor([]).to(device)

    for model in models:
        params_init_net = util.flatten(model).to(device).clone()
        params_init = torch.cat((params_init, params_init_net))

    # params_init = torch.randn_like(params_init)
    if params_init_val is not None:
        params_init = params_init_val

    log_prob_func = define_model_log_prob_bpinns(models, model_loss, data, tau_priors, tau_likes,
                                                 n_params_single=n_params_single, pde=pde)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_time = time.time()
    if pinns:
        params = params_init.clone().detach().requires_grad_()
        optimizer = torch.optim.Adam([params], lr=step_size)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = - log_prob_func(params)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print('Epoch: %d, loss: %.6f, time: %.2f' % (epoch, loss.detach().item(), time.time() - start_time))

        if not store_on_GPU:
            ret_params = [params.clone().detach().cpu()]
        else:
            ret_params = [params.clone()]

        return list(map(lambda t: t.detach(), ret_params))

    else:
        return Bayesian.sample(log_prob_func, params_init, num_samples=num_samples,
                               num_steps_per_sample=num_steps_per_sample, step_size=step_size, burn=burn,
                               inv_mass=inv_mass, normalizing_const=normalizing_const, sampler=sampler,
                               integrator=integrator, metric=metric, debug=debug, store_on_GPU=store_on_GPU)


def predict_model_bpinns(models, samples, data, model_loss, tau_priors=None, tau_likes=None, n_params_single=None,
                         pde=False):
    """
    预测过程
    """

    if pde:

        log_prob_func = define_model_log_prob_bpinns(models, model_loss, data, tau_priors, tau_likes, predict=True,
                                                     n_params_single=n_params_single, pde=pde)

        pred_log_prob_list = []
        pred_list = []
        _, pred = log_prob_func(samples[0])
        for i in range(len(pred)):
            pred_list.append([])

        for s in samples:
            lp, pred = log_prob_func(s)
            pred_log_prob_list.append(lp.detach())  # Side effect is to update weights to be s
            for i in range(len(pred_list)):
                pred_list[i].append(pred[i].detach())

        for i in range(len(pred_list)):
            pred_list[i] = torch.stack(pred_list[i])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pred_list, pred_log_prob_list

    else:
        with torch.no_grad():

            log_prob_func = define_model_log_prob_bpinns(models, model_loss, data, tau_priors, tau_likes, predict=True,
                                                         n_params_single=n_params_single, pde=pde)

            pred_log_prob_list = []
            pred_list = []
            _, pred = log_prob_func(samples[0])
            for i in range(len(pred)):
                pred_list.append([])

            for s in samples:
                lp, pred = log_prob_func(s)
                pred_log_prob_list.append(lp.detach())  # Side effect is to update weights to be s
                for i in range(len(pred_list)):
                    pred_list[i].append(pred[i].detach())

            for i in range(len(pred_list)):
                pred_list[i] = torch.stack(pred_list[i])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return pred_list, pred_log_prob_list
