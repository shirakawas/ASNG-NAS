import csv
import time
import numpy as np
import torch
from torch import nn

from common.eval_test import evaluate


def arch_search_valid(model, train_data, test_data, corrupt_func, optimizer, lr_scheduler, clip_value=1., batchsize=16,
                      lam=2, valid_rate=0.5, gpu_id=0, period=None, out_model='out_model.model', log_file='log.csv'):

    if period is None:
        period = {'max_ite': 200000, 'save': 4000, 'verbose_ite': 100}

    start = time.time()

    if gpu_id >= 0:
        model = model.cuda(gpu_id)
    loss_func = nn.MSELoss()

    # Data loader
    valid_size = int(valid_rate * len(train_data))
    print('#(train / valid / test) = (%d, %d, %d)' % (len(train_data) - valid_size, valid_size, len(test_data)))

    inds = list(range(len(train_data)))
    np.random.shuffle(inds)
    train_loader = torch.utils.data.DataLoader(train_data, batchsize,
                                               sampler=torch.utils.data.SubsetRandomSampler(inds[:-valid_size]))
    valid_loader = torch.utils.data.DataLoader(train_data, batchsize,
                                               sampler=torch.utils.data.SubsetRandomSampler(inds[-valid_size:]))

    # log header
    with open(log_file, 'w') as fp:
        writer = csv.writer(fp, lineterminator='\n')
        header_list = ['epoch', 'iteration', 'train_time', 'lr', 'train_loss', 'valid_loss', 'test_MSE', 'test_PSNR',
                       'test_SSIM', 'model_params']
        header_list += model.asng.log_header(theta_log=True)
        writer.writerow(header_list)

    train_time = train_loss = valid_loss = 0.
    epoch = ite = n = 0
    losses = np.zeros(lam)
    M = np.zeros((lam, model.asng.d, model.asng.Cmax))

    while ite < period['max_ite']:
        epoch += 1
        for train_batch, valid_batch in zip(train_loader, valid_loader):
            ite_start = time.time()
            ite += 1

            # ---------- One iteration of the training loop ----------
            model.train()
            lr_scheduler.step()

            X, _ = train_batch
            Xv, _ = valid_batch
            if gpu_id >= 0:
                X = X.cuda(gpu_id)
                Xv = Xv.cuda(gpu_id)

            optimizer.zero_grad()  # Clear gradient
            loss_mean = 0.

            # Update weights
            for i in range(lam):
                # Calculate the prediction of the network
                Y, _ = model(corrupt_func(X), stochastic=True)
                loss = loss_func(Y, X)  # Calculate the MSE loss
                loss_mean += loss / lam
                train_loss += loss.item()
            loss_mean.backward()  # Calculate the gradient
            del loss_mean
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # Gradient clipping
            optimizer.step()  # Update the trainable parameters

            # Update theta
            with torch.no_grad():
                for i in range(lam):
                    Y, M[i] = model(corrupt_func(Xv), stochastic=True)
                    loss = loss_func(Y, Xv).item()
                    valid_loss += loss
                    losses[i] = loss
                model.p_model_update(M, losses)

            n += lam
            train_time += time.time() - ite_start
            # --------------------- until here ---------------------

            is_save = ite % period['save'] == 0
            if ite % period['verbose_ite'] == 0 or is_save or ite == period['max_ite'] or ite == 1:
                # Display the training loss
                print('epoch:{} iteration:{} elapse_time:{:.04f} lr:{:e} cur_train_loss:{:.04f} cur_valid_loss:{:.04f} delta:{} theta_converge:{:.04f}'
                      .format(epoch, ite, (time.time() - start) / 60, lr_scheduler.get_lr()[0],
                              train_loss / n, valid_loss / n, model.asng.delta, model.asng.theta.max(axis=1).mean()))

            # Check the test loss
            if is_save or ite == period['max_ite']:
                # Testing
                test_res = evaluate(model, test_data, corrupt_func, gpu_id=gpu_id, batchsize=batchsize)
                max_params = np.sum(np.prod(param.size()) for param in model.parameters())
                model_params = model.get_params_mle()
                print('test_MSE:{:.04f} test_PSNR:{:.04f} test_SSIM:{:.04f} param_num:{} param_ratio:{:.04f} active_num:{}'.
                      format(test_res['MLE_MSE'], test_res['MLE_PSNR'], test_res['MLE_SSIM'], model_params,
                             model_params / max_params, int(model.is_active.sum())))
                print(model.mle_network_string(sep=' ') + '\n')

                # Save log
                with open(log_file, 'a') as fp:
                    writer = csv.writer(fp, lineterminator='\n')
                    log_list = [epoch, ite, train_time, lr_scheduler.get_lr()[0], train_loss / n, valid_loss / n,
                                test_res['MLE_MSE'], test_res['MLE_PSNR'], test_res['MLE_SSIM'], model_params]
                    log_list += model.asng.log(theta_log=True)
                    writer.writerow(log_list)

                train_loss = valid_loss = 0.
                n = 0
                if ite >= period['max_ite']:
                    break

    # Save model
    if out_model is not None:
        torch.save(model.state_dict(), out_model)

    return model, train_time


def train(model, train_data, test_data, corrupt_func, optimizer, lr_scheduler, clip_value=1., batchsize=16,
          gpu_id=0, period=None, out_model='out_model.model', log_file='log.csv'):

    if period is None:
        period = {'max_ite': 200000, 'save': 4000, 'verbose_ite': 100}

    start = time.time()

    if gpu_id >= 0:
        model = model.cuda(gpu_id)
    loss_func = nn.MSELoss()

    M = model.asng.mle()

    # Data loader
    train_loader = torch.utils.data.DataLoader(train_data, batchsize, shuffle=True, drop_last=False)

    # log header
    with open(log_file, 'w') as fp:
        writer = csv.writer(fp, lineterminator='\n')
        header_list = ['epoch', 'iteration', 'train_time', 'lr', 'train_loss', 'test_MSE', 'test_PSNR', 'test_SSIM',
                       'model_params']
        writer.writerow(header_list)

    train_time = train_loss = 0.
    epoch = ite = n = 0

    while ite < period['max_ite']:
        epoch += 1
        for X, _ in train_loader:
            ite_start = time.time()
            ite += 1
            # ---------- One iteration of the training loop ----------
            model.train()
            lr_scheduler.step()

            if gpu_id >= 0:
                X = X.cuda(gpu_id)

            optimizer.zero_grad()  # Clear gradient

            # Calculate the prediction of the network
            Y = model.forward_as(M, corrupt_func(X))
            loss = loss_func(Y, X)  # Calculate the MSE loss
            train_loss += loss.item()
            n += 1

            loss.backward()  # Calculate the gradient
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # Gradient clipping
            optimizer.step()  # Update the trainable parameters

            train_time += time.time() - ite_start
            # --------------------- until here ---------------------

            is_save = ite % period['save'] == 0
            if ite % period['verbose_ite'] == 0 or is_save or ite == period['max_ite'] or ite == 1:
                # Display the training loss
                print('epoch:{} iteration:{} elapse_time:{:.04f} lr:{:e} cur_loss:{:.04f}'
                      .format(epoch, ite, (time.time() - start) / 60, lr_scheduler.get_lr()[0], loss))

            # Check the test loss
            if is_save or ite == period['max_ite']:
                # Testing
                test_res = evaluate(model, test_data, corrupt_func, gpu_id=gpu_id, batchsize=batchsize)
                model_params = model.get_params_mle()
                print('test_MSE:{:.04f} test_PSNR:{:.04f} test_SSIM:{:.04f} param_num:{} active_num:{}'.
                      format(test_res['MLE_MSE'], test_res['MLE_PSNR'], test_res['MLE_SSIM'], model_params,
                             int(model.is_active.sum())))
                print(model.mle_network_string(sep=' ') + '\n')

                # Save log
                with open(log_file, 'a') as fp:
                    writer = csv.writer(fp, lineterminator='\n')
                    log_list = [epoch, ite, train_time, lr_scheduler.get_lr()[0], train_loss / n, test_res['MLE_MSE'],
                                test_res['MLE_PSNR'], test_res['MLE_SSIM'], model_params]
                    writer.writerow(log_list)

                train_loss = 0.
                n = 0
                if ite >= period['max_ite']:
                    break

    # Save model
    if out_model is not None:
        torch.save(model.state_dict(), out_model)

    return model, train_time
