#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime

import classification.operations as O
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision
from classification import utils
from classification.cnn_model_micro import MicroProbabilisticCNN
from classification.train import train, architecture_search

n_classes = 10
params_limit = 4.0


def experiment(model_class, ops_server_class, initial_lr, momentum, weight_decay, nesterov, cutout, n_valid_data, n_epochs_search, n_epochs_retrain,
               batch_size_search, batch_size_retrain, lam, max_droppath_rate, grad_clip_value_search, grad_clip_value_retrain, weight_auxiliary,
               alpha, init_delta, gpu_id, seed=None, log_file_header='', count_sample=False, log_asng=False, minus_test_time=False):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    log_dir = os.path.join(log_file_header + datetime.now().strftime('%Y%m%d%H%M%S'), '')
    print('log directory:', log_dir)
    os.mkdir(log_dir)

    with open(os.path.join(log_dir, 'description.txt'), 'w') as o:
        o.write('class: %s\n' % model_class.__name__)
        o.write('ops: %s\n' % ops_server_class.__name__)
        o.write('initial_lr: %f\n' % initial_lr)
        o.write('momentum: %f\n' % momentum)
        o.write('weight_decay: %f\n' % weight_decay)
        o.write('nesterov: %s\n' % ('True' if nesterov else 'False'))
        o.write('cutout: %s\n' % ('True' if cutout else 'False'))
        o.write('n_valid_data: %d\n' % n_valid_data)
        o.write('n_epochs_search: %d\n' % n_epochs_search)
        o.write('n_epochs_retrain: %d\n' % n_epochs_retrain)
        o.write('batch_size_search: %d\n' % batch_size_search)
        o.write('batch_size_retrain: %d\n' % batch_size_retrain)
        o.write('lam: %d\n' % lam)
        o.write('max_droppath_rate: %f\n' % max_droppath_rate)
        o.write('grad_clip_value_search: %f\n' % grad_clip_value_search)
        o.write('grad_clip_value_retrain: %f\n' % grad_clip_value_retrain)
        o.write('weight_auxiliary: %f\n' % weight_auxiliary)
        o.write('alpha: %f\n' % alpha)
        o.write('init_delta: %f\n' % init_delta)
        o.write('minus_test_time: %s\n' % ('True' if minus_test_time else 'False'))
        if seed is not None:
            o.write('seed: %d\n' % seed)
        else:
            o.write('seed: random\n')

    if gpu_id >= 0:
        torch.cuda.set_device(gpu_id)
        cudnn.benchmark = True
        cudnn.enabled = True 

    loss_func = nn.CrossEntropyLoss()
    if gpu_id >= 0:
        loss_func = loss_func.cuda()

    train_transform, test_transform = utils._data_transforms_cifar10(cutout=cutout)
    train_data = torchvision.datasets.CIFAR10('./data', train=True, transform=train_transform, download=True)
    test_data = torchvision.datasets.CIFAR10('./data', train=False, transform=test_transform, download=True)

    print('Architecture Search...')
    nn_model = model_class(
            n_classes=n_classes,
            n_cells=8,
            ops_server=ops_server_class(),
            alpha=alpha,
            init_delta=init_delta,
        )
    print('dim theta: %d' % nn_model.p_model.C.sum())
    if gpu_id >= 0:
        nn_model = nn_model.cuda()
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs_search)

    nn_model = architecture_search(
            nn_model=nn_model,
            loss_func=loss_func,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            n_epochs=n_epochs_search,
            batch_size=batch_size_search,
            clip_value=grad_clip_value_search,
            train_data=train_data,
            test_data=test_data,
            n_valid_data=n_valid_data,
            lam=lam,
            gpu_id=gpu_id,
            log_file=os.path.join(log_dir, 'search_log.csv'),
            cntlog_file=os.path.join(log_dir, 'search_cnt_log.csv') if count_sample else None,
            asnglog_file=os.path.join(log_dir, 'search_asng_log.csv') if log_asng else None,
            minus_test_time=minus_test_time
        )

    with open(os.path.join(log_dir, 'trained_theta.csv'), 'w') as o:
        col = ['dim'] + [str(i) for i in range(nn_model.p_model.Cmax)]
        for i, c in enumerate(col):
            if i != 0:
                o.write(',')
            o.write(c)
        o.write('\n')
        for i, r in enumerate(nn_model.p_model.theta):
            o.write(str(i))
            for c in r:
                o.write(',')
                o.write(str(c))
            o.write('\n')

    # Binary search to find channels_init.
    trained_theta = nn_model.p_model.theta
    lo, hi = -1, 128
    while hi - lo > 1:
        mid = lo + (hi - lo)//2
        nn_model = model_class(
                n_classes=n_classes,
                n_cells=20,
                affine=True,
                ops_server=ops_server_class(affine=True),
                trained_theta=trained_theta,
                channels_init=mid
            )
        n_params = nn_model.get_params_mle()
        if n_params/1e6 <= params_limit:
            lo = mid
        else:
            hi = mid
    channels_init = lo
    print('channels_init: %d' % channels_init)

    print('Retraining...')
    nn_model = model_class(
            n_classes=n_classes,
            n_cells=20,
            channels_init=channels_init,
            affine=True,
            ops_server=ops_server_class(affine=True),
            trained_theta=trained_theta
        )
    nn_model.fix_arc()
    print('n_params: %fM' % (nn_model.get_params_mle()/1e6))

    with open(os.path.join(log_dir, 'description.txt'), 'a') as o:
        o.write('dim_theta: %d\n' % nn_model.p_model.C.sum())
        o.write('channels_init: %d\n' % channels_init)
        o.write('n_params: %fM\n' % (nn_model.get_params_mle()/1e6))

    if gpu_id >= 0:
        nn_model = nn_model.cuda()
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs_retrain)
    nn_model = train(
            nn_model=nn_model,
            loss_func=loss_func,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            n_epochs=n_epochs_retrain,
            batch_size=batch_size_retrain,
            max_droppath_rate=max_droppath_rate,
            clip_value=grad_clip_value_retrain,
            weight_auxiliary=weight_auxiliary,
            train_data=train_data,
            test_data=test_data,
            gpu_id=gpu_id,
            log_file=os.path.join(log_dir, 'train_log.csv')
        )

    torch.save(nn_model.state_dict(), os.path.join(log_dir, 'trained_model.pt'))


if __name__ == '__main__':
    experiment(
        model_class=MicroProbabilisticCNN,
        ops_server_class=O.ENASOpsServer,
        initial_lr=0.025,
        momentum=0.9,
        weight_decay=3e-4,
        nesterov=False,
        cutout=True,
        n_valid_data=25000,
        n_epochs_search=100,
        n_epochs_retrain=600,
        batch_size_search=64,
        batch_size_retrain=80,
        lam=2,
        max_droppath_rate=0.3,
        grad_clip_value_search=5,
        grad_clip_value_retrain=5,
        weight_auxiliary=0.4,
        alpha=1.5,
        init_delta=1.0,
        gpu_id=0,
        seed=np.random.randint(10000),
        log_file_header='arc_search_classification',
        count_sample=True,
        log_asng=True,
        minus_test_time=True
    )