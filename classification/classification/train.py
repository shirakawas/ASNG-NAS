#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import numpy as np
import torch
from torch import nn

from classification import utils


def write_row(filename, data, reset=False):
    with open(filename, 'w' if reset else 'a') as o:
        row = ''
        for i, c in enumerate(data):
            row += ('' if i == 0 else ',') + str(c)
        row += '\n'
        o.write(row)

def architecture_search(nn_model, loss_func, optimizer, lr_scheduler, n_epochs, batch_size,
                        clip_value, train_data, test_data=None, n_valid_data=5000, lam=2, gpu_id=-1,
                        minus_test_time=False, log_file='search_log.csv', cntlog_file='search_cnt_log.csv', asnglog_file='search_asng_log.csv'):

    inds = list(range(len(train_data)))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, sampler=torch.utils.data.SubsetRandomSampler(inds[:-n_valid_data]), num_workers=1)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size, sampler=torch.utils.data.SubsetRandomSampler(inds[-n_valid_data:]), num_workers=1)

    len_train = len(train_data) - n_valid_data

    write_row(log_file, ['epoch', 'elapsed_time', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc', 'test_loss', 'test_acc', 'theta_convergence', 'lr'], reset=True)

    if cntlog_file is not None:
        write_row(
            cntlog_file,
            ['epoch']
            + ['cnt%d_%d_%d' % (i, j, k) for i in range(nn_model.n_nodes) for j in range(-2, nn_model.n_nodes) for k in range(nn_model.n_ops)]
            + ['theta%d_%d' % (i, j) for i in range(len(nn_model.p_model.C)) for j in range(nn_model.p_model.Cmax)],
            reset=True
        )
        cnt = np.zeros((nn_model.n_nodes, nn_model.n_nodes + 2, nn_model.n_ops), dtype='int32')

    if asnglog_file is not None:
        write_row(asnglog_file, ['iteration', 'Delta', 's^2', 'gamma', 'lam', 'delta', 'alpha'], reset=True)
        iteration = 0

    start_time = time.time()
    for epoch in range(n_epochs):
        nn_model.train()
        lr_scheduler.step()

        train_loss = 0.
        train_acc = 0.
        valid_loss = 0.
        valid_acc = 0.

        for (X, t), (X_valid, t_valid) in zip(train_loader, valid_loader):
            if gpu_id >= 0:
                X, t = X.cuda(gpu_id), t.cuda(gpu_id)
                X_valid, t_valid = X_valid.cuda(gpu_id), t_valid.cuda(gpu_id)

            loss_sum = 0.
            optimizer.zero_grad()
            for _ in range(lam):
                h, M, _ = nn_model(X, True)
                loss = loss_func(h, t)
                loss_sum = loss_sum + loss/lam

                train_loss += loss.item() * len(X)/(len_train*lam)
                train_acc += utils.accuracy(h, t, topk=(1,))[0].item() * len(X)/(len_train*lam)

                if cntlog_file is not None:
                    # count selected operations.
                    M_dec = M.argmax(axis=1)
                    for i in range(nn_model.n_nodes):
                        ind = 4*i
                        cnt[i, M_dec[ind], M_dec[ind + 2]] += 1
                        cnt[i, M_dec[ind + 1], M_dec[ind + 3]] += 1


            loss_sum.backward()
            nn.utils.clip_grad_norm_(nn_model.parameters(), clip_value)
            optimizer.step()

            Ms = []
            losses = []
            with torch.no_grad():
                for _ in range(lam):
                    h_valid, M, _ = nn_model(X_valid, True)
                    loss = loss_func(h_valid, t_valid)

                    valid_loss += loss.item() * len(X)/(len_train*lam)
                    valid_acc += utils.accuracy(h_valid, t_valid, topk=(1,))[0].item() * len(X)/(len_train*lam)

                    Ms.append(M)
                    losses.append(loss.item())

            losses, Ms = np.array(losses), np.array(Ms)
            nn_model.p_model_update(Ms, losses)


            if asnglog_file is not None:
                ss = (nn_model.asng.s ** 2).sum()
                write_row(asnglog_file, [iteration, nn_model.asng.Delta, ss, nn_model.asng.gamma, 2, nn_model.asng.get_delta(), ss/nn_model.asng.gamma])
                iteration += 1

        test_loss = test_acc = np.nan
        if test_data is not None:
            if minus_test_time:
                test_start = time.time()
            test_loss, test_acc = test(nn_model=nn_model, loss_func=loss_func, test_data=test_data, batch_size=batch_size, gpu_id=gpu_id)
            if minus_test_time:
                start_time += time.time() - test_start

        elapsed_time = time.time() - start_time
        convergence = nn_model.p_model.theta.max(axis=1).mean()
        print('epoch: %d, elapsed_time: %f, train_loss: %f, train_acc: %f, valid_loss: %f, valid_acc: %f, test_loss: %f, test_acc: %f, convergence: %f, lr: %f'
            % (epoch, elapsed_time, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc, convergence, lr_scheduler.get_lr()[0]))
        write_row(log_file, [epoch, elapsed_time, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc, convergence, lr_scheduler.get_lr()[0]])

        if cntlog_file is not None:
            write_row(cntlog_file, [epoch] + list(cnt.reshape(-1)) + list(nn_model.p_model.theta.reshape(-1)))

    return nn_model


def train(nn_model, loss_func, optimizer, lr_scheduler, n_epochs, batch_size,
          max_droppath_rate, clip_value, weight_auxiliary, train_data, test_data=None,gpu_id=-1, log_file='train_log.csv'):
    """Function for deterministic training.

    The most likely architecture is used.
    """

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=2)

    write_row(log_file, ['epoch', 'elapsed_time', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'theta_convergence', 'lr'], reset=True)

    start_time = time.time()
    for epoch in range(n_epochs):
        nn_model.train()
        lr_scheduler.step()
        cur_droppath_rate = max_droppath_rate * epoch/n_epochs

        train_loss = 0.
        train_acc = 0.

        for X, t in train_loader:
            if gpu_id >= 0:
                X, t = X.cuda(gpu_id), t.cuda(gpu_id)

            optimizer.zero_grad()
            h, _, h_auxiliary = nn_model(X, False, cur_droppath_rate)
            loss_main = loss_func(h, t)
            loss_auxiliary = loss_func(h_auxiliary, t)
            loss = loss_main + weight_auxiliary*loss_auxiliary
            loss.backward()
            nn.utils.clip_grad_norm_(nn_model.parameters(), clip_value)
            optimizer.step()

            train_loss += loss_main.item() * len(X)/len(train_data)
            train_acc += utils.accuracy(h, t, topk=(1,))[0].item() * len(X)/len(train_data)

        test_loss = test_acc = np.nan
        if test_data is not None:
            test_loss, test_acc = test(nn_model, loss_func, test_data, batch_size, gpu_id)
        elapsed_time = time.time() - start_time
        convergence = nn_model.p_model.theta.max(axis=1).mean()
        print('epoch: %d, elapsed_time: %f, train_loss: %f, train_acc: %f, test_loss: %f, test_acc: %f, convergence: %f, lr: %f'
            % (epoch, elapsed_time, train_loss, train_acc, test_loss, test_acc, convergence, lr_scheduler.get_lr()[0]))
        write_row(log_file, [epoch, elapsed_time, train_loss, train_acc, test_loss, test_acc, convergence, lr_scheduler.get_lr()[0]])

    return nn_model



def test(nn_model, loss_func, test_data, batch_size, gpu_id=-1):
    """Function for deterministic evaluation.

    The most likely architecture is used.
    """
    nn_model.eval()

    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, num_workers=2)

    loss_avg = 0.
    acc_avg = 0.

    with torch.no_grad():
        for X, t in test_loader:
            if gpu_id >= 0:
                X, t = X.cuda(gpu_id), t.cuda(gpu_id)

            h, _, _ = nn_model(X, False)
            loss_avg += loss_func(h, t).item() * len(X)/len(test_data)
            acc_avg += utils.accuracy(h, t, topk=(1,))[0].item() * len(X)/len(test_data)

    return loss_avg, acc_avg



