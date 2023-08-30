#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/7/16 23:40
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : NeuralOperators.py
# @Description    : ******
"""

import os
import numpy as np
import torch
import time
from shutil import copyfile
from Models.basic._base_module import BasicModule


class NeuralOperators(BasicModule):

    def __init__(self,
                 network,
                 device,
                 lossfunc,
                 optimizer,
                 scheduler,
                 work_space,
                 data_normer,
                 **kwargs):
        super(NeuralOperators, self).__init__()
        self.name = 'NeuralOperators'


        self.device = device
        self.network = network.to(self.device)
        self.work_space = work_space
        self.data_normer = data_normer
        self.work_space = work_space

        self._set_lossfunc(lossfunc)
        self._set_optimizer(optimizer)
        self._set_scheduler(scheduler)

        self.metric = kwargs.get('metric', None)
        self.save_epoch_freq = kwargs.get('save_epoch_freq', 10)
        self.disp_epoch_freq = kwargs.get('disp_epoch_freq', 1)
        self.lowest_loss = np.inf

        self.history_loss = {'train': [], 'valid': []}
        self.history_metric = {'train': [], 'valid': []}
    def train(self, trainloader, validloader, epochs, **kwargs):


        for epoch in range(1, epochs+1):

            star_time = time.time()
            # train
            self.network.train()
            for (input, output, target) in trainloader:
                self._train_batch(input.data.to(self.device),
                                  output.data.to(self.device),
                                  grid=output.grid.to(self.device),
                                  lossfunc=self.lossfunc)

            if epoch % self.disp_epoch_freq == 0:
                # train_loss logger, # valid_loss logger
                train_loss = []
                valid_loss = []
                self.network.eval()
                for (input, output, target) in trainloader:
                    loss_batch = self._valid_batch(input.data.to(self.device),
                                                   output.data.to(self.device),
                                                   grid=output.grid.to(self.device),
                                                   lossfunc=self.lossfunc)
                    train_loss.append(loss_batch)
                for (input, output, target) in validloader:
                    loss_batch = self._valid_batch(input.data.to(self.device),
                                                   output.data.to(self.device),
                                                   grid=output.grid.to(self.device),
                                                   lossfunc=self.lossfunc)
                    valid_loss.append(loss_batch)

                self.history_loss['train'].append(np.array(train_loss, dtype=np.float32))
                self.history_loss['valid'].append(np.array(valid_loss, dtype=np.float32))

                self.work_space.logger.info('epoch: {:5d}, lr: {:.3e}, cost: {:.3f},  '
                                            'train_fields_loss: {:.3e}, valid_fields_loss: {:.3e} '.
                                            format(epoch,
                                                   self.optimizer.param_groups[0]['lr'], time.time() - star_time,
                                                   self.history_loss['train'][-1].mean(),
                                                   self.history_loss['valid'][-1].mean(),
                                                   ))

            if epoch % self.save_epoch_freq == 0:

                save_file = os.path.join(self.work_space.train_path, 'epoch_' + str(epoch) + '.pth')
                torch.save({
                            'net_model': self.network.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                            'history_loss': self.history_loss,},
                           save_file)
                copyfile(save_file, os.path.join(self.work_space.train_path, 'latest_model.pth'))

                if self.history_loss['valid'][-1].mean() < self.lowest_loss:
                    self.lowest_loss = self.history_loss['valid'][-1].mean()
                    copyfile(save_file, os.path.join(self.work_space.train_path, 'lowest_model.pth'))

            if epoch in self.scheduler.milestones:
                checkpoint = torch.load(os.path.join(self.work_space.train_path, 'lowest_model.pth'))
                self.network.load_state_dict(checkpoint['net_model'])
                # self.optimizer.load_state_dict(checkpoint['optimizer'])
                # self.scheduler.load_state_dict(checkpoint['scheduler'])
                # self.history_loss = checkpoint['history_loss']
                self.work_space.logger.warning("model load successful!")

            # scheduler
            self.scheduler.step()

    def infer(self, input, grid=None, mesh=None, edge=None, **kwargs):

        if input is not torch.Tensor:
            input = torch.tensor(input, dtype=torch.float32, device=self.device)

        if grid is not None and grid is not torch.Tensor:
            grid = torch.tensor(grid, dtype=torch.float32, device=self.device)

        if mesh is not None and mesh is not torch.Tensor:
            mesh = torch.tensor(mesh, dtype=torch.float32, device=self.device)

        if edge is not None and edge is not torch.Tensor:
            edge = torch.tensor(edge, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.data_normer is not None:
                input = self.data_normer['input'].norm(input)
                if grid is not None:
                    grid = self.data_normer['grid'].norm(grid)
                if mesh is not None:
                    mesh = self.data_normer['mesh'].norm(mesh)
            output = self.network(input.to(self.device),
                                  grid=grid,
                                  mesh=mesh,
                                  edge=edge)
            if self.data_normer is not None:
                output = self.data_normer['output'].back(output)
            return output.cpu().numpy()

    def _valid_batch(self, input, output, lossfunc, grid=None, mesh=None, edge=None, **kwargs):

        with torch.no_grad():
            if self.data_normer is not None:
                input = self.data_normer['input'].norm(input)
                output = self.data_normer['output'].norm(output)
                if grid is not None:
                    grid = self.data_normer['grid'].norm(grid)
                if mesh is not None:
                    mesh = self.data_normer['mesh'].norm(mesh)

            _output = self.network(input, grid=grid, mesh=mesh, edge=edge)
            _loss = lossfunc(_output, output)
            if self.metric is not None:
                _metric = self.metric(_output, output)
                return _loss.item(), _metric
            else:
                return _loss.item()

    def _train_batch(self, input, output, lossfunc, grid=None, mesh=None, edge=None, **kwargs):

        if self.data_normer is not None:
            input = self.data_normer['input'].norm(input)
            output = self.data_normer['output'].norm(output)
            if grid is not None:
                grid = self.data_normer['grid'].norm(grid)
            if mesh is not None:
                mesh = self.data_normer['mesh'].norm(mesh)

        _output = self.network(input, grid=grid, mesh=mesh, edge=edge)
        _loss = lossfunc(_output, output)
        self.optimizer.zero_grad()
        _loss.backward()
        self.optimizer.step()
        return _loss.item()

def run_train(model, loss_func, metric_func,
              train_loader, valid_loader,
              optimizer, lr_scheduler,
              train_batch=None,
              validate_epoch=None,
              epochs=10,
              device="cuda",
              mode='min',
              tqdm_mode='batch',
              patience=10,
              grad_clip=0.999,
              start_epoch: int = 0,
              model_save_path=MODEL_PATH,
              save_mode='state_dict',  # 'state_dict' or 'entire'
              model_name='model.pt',
              result_name='result.pt'):
    loss_train = []
    loss_val = []
    loss_epoch = []
    lr_history = []
    it = 0

    if patience is None or patience == 0:
        patience = epochs
    start_epoch = start_epoch
    end_epoch = start_epoch + epochs
    best_val_metric = -np.inf if mode == 'max' else np.inf
    best_val_epoch = None
    save_mode = 'state_dict' if save_mode is None else save_mode
    stop_counter = 0
    is_epoch_scheduler = any(s in str(lr_scheduler.__class__)
                             for s in EPOCH_SCHEDULERS)
    tqdm_epoch = False if tqdm_mode == 'batch' else True

    with tqdm(total=end_epoch-start_epoch, disable=not tqdm_epoch) as pbar_ep:
        for epoch in range(start_epoch, end_epoch):
            model.train()
            with tqdm(total=len(train_loader), disable=tqdm_epoch) as pbar_batch:
                for batch in train_loader:
                    if is_epoch_scheduler:
                        loss, _, _ = train_batch(model, loss_func,
                                                 batch, optimizer,
                                                 None, device, grad_clip=grad_clip)
                    else:
                        loss, _, _ = train_batch(model, loss_func,
                                                 batch, optimizer,
                                                 lr_scheduler, device, grad_clip=grad_clip)
                    loss = np.array(loss)
                    loss_epoch.append(loss)
                    it += 1
                    lr = optimizer.param_groups[0]['lr']
                    lr_history.append(lr)
                    desc = f"epoch: [{epoch+1}/{end_epoch}]"
                    if loss.ndim == 0:  # 1 target loss
                        _loss_mean = np.mean(loss_epoch)
                        desc += f" loss: {_loss_mean:.3e}"
                    else:
                        _loss_mean = np.mean(loss_epoch, axis=0)
                        for j in range(len(_loss_mean)):
                            if _loss_mean[j] > 0:
                                desc += f" | loss {j}: {_loss_mean[j]:.3e}"
                    desc += f" | current lr: {lr:.3e}"
                    pbar_batch.set_description(desc)
                    pbar_batch.update()

            loss_train.append(_loss_mean)
            # loss_train.append(loss_epoch)
            loss_epoch = []

            val_result = validate_epoch(
                model, metric_func, valid_loader, device)

            loss_val.append(val_result["metric"])
            val_metric = val_result["metric"].sum()
            if mode == 'max':
                if val_metric > best_val_metric:
                    best_val_epoch = epoch
                    best_val_metric = val_metric
                    stop_counter = 0
                else:
                    stop_counter += 1
            else:
                if val_metric < best_val_metric:
                    best_val_epoch = epoch
                    best_val_metric = val_metric
                    stop_counter = 0
                    if save_mode == 'state_dict':
                        torch.save(model.state_dict(), os.path.join(
                            model_save_path, model_name))
                    else:
                        torch.save(model, os.path.join(
                            model_save_path, model_name))
                    best_model_state_dict = {
                        k: v.to('cpu') for k, v in model.state_dict().items()}
                    best_model_state_dict = OrderedDict(best_model_state_dict)

                else:
                    stop_counter += 1

            if lr_scheduler and is_epoch_scheduler:
                if 'ReduceLROnPlateau' in str(lr_scheduler.__class__):
                    lr_scheduler.step(val_metric)
                else:
                    lr_scheduler.step()

            if stop_counter > patience:
                print(f"Early stop at epoch {epoch}")
                break
            if val_result["metric"].ndim == 0:
                desc = color(
                    f"| val metric: {val_metric:.3e} ", color=Colors.blue)
            else:
                metric_0, metric_1 = val_result["metric"][0], val_result["metric"][1]
                desc = color(
                    f"| val metric 1: {metric_0:.3e} ", color=Colors.blue)
                desc += color(f"| val metric 2: {metric_1:.3e} ",
                              color=Colors.blue)
            desc += color(
                f"| best val: {best_val_metric:.3e} at epoch {best_val_epoch+1}", color=Colors.yellow)
            desc += color(f" | early stop: {stop_counter} ", color=Colors.red)
            desc += color(f" | current lr: {lr:.3e}", color=Colors.magenta)
            if not tqdm_epoch:
                tqdm.write("\n"+desc+"\n")
            else:
                desc_ep = color("", color=Colors.green)
                if _loss_mean.ndim == 0:  # 1 target loss
                    desc_ep += color(f"| loss: {_loss_mean:.3e} ",
                                     color=Colors.green)
                else:
                    for j in range(len(_loss_mean)):
                        if _loss_mean[j] > 0:
                            desc_ep += color(
                                f"| loss {j}: {_loss_mean[j]:.3e} ", color=Colors.green)
                desc_ep += desc
                pbar_ep.set_description(desc_ep)
                pbar_ep.update()

            result = dict(
                best_val_epoch=best_val_epoch,
                best_val_metric=best_val_metric,
                loss_train=np.asarray(loss_train),
                loss_val=np.asarray(loss_val),
                lr_history=np.asarray(lr_history),
                # best_model=best_model_state_dict,
                optimizer_state=optimizer.state_dict()
            )
            save_pickle(result, os.path.join(model_save_path, result_name))
    return result
