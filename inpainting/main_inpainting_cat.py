#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import numpy as np

import torch
import torch.utils
import torchvision
import torch.backends.cudnn as cudnn

from common import utils
from common.utils import RandomPixelMasking, RandomHalfMasking, CenterMasking
from common.eval_test import evaluate
from inpainting_cat.cae_model import ProbablisticCAE
from inpainting_cat.train import arch_search_valid, train


def load_data(path='../data/', data_name='celebA', img_size=64):
    print('Loading ' + data_name + 'data...')
    train_transform, test_transform = utils.data_transforms(img_size=img_size)

    if data_name != 'svhn':
        # The image data should be contained in sub folders (e.g., ../data/celebA/train/image/aaa.png)
        train_data = torchvision.datasets.ImageFolder('{}{}/train'.format(path, data_name), transform=train_transform)
        test_data = torchvision.datasets.ImageFolder('{}{}/test'.format(path, data_name), transform=test_transform)
    else:
        train_data = torchvision.datasets.SVHN(path, split='train', transform=train_transform, download=True)
        test_data = torchvision.datasets.SVHN(path, split='test', transform=test_transform, download=True)
        # extra_data = torchvision.datasets.SVHN(path, split='extra', transform=train_transform, download=True)
        # train_data = torch.utils.data.ConcatDataset([train_data, extra_data])

    print('train_data_size: %d, test_data_size: %d' % (len(train_data), len(test_data)))
    return train_data, test_data


# Save result data
class SaveResult(object):
    def __init__(self, res_file_name='result.csv'):
        self.res_file_name = res_file_name
        # header
        with open(self.res_file_name, 'w') as fp:
            writer = csv.writer(fp, lineterminator='\n')
            writer.writerow(['exp_index', 'train_time', 'MLE_MSE', 'MLE_PSNR', 'MLE_SSIM', 'det_param', 'max_param',
                             'node_num', 'cat_d', 'cat_valid_d', 'cat_param_num', 'active_num', 'net_str'])

    def save(self, exp_index, model, train_time, res):
        dist = model.asng
        params = np.sum(np.prod(param.size()) for param in model.parameters())
        net_str = model.mle_network_string(sep=' ')
        with open(self.res_file_name, 'a') as fp:
            writer = csv.writer(fp, lineterminator='\n')
            writer.writerow([exp_index, train_time, res['MLE_MSE'], res['MLE_PSNR'], res['MLE_SSIM'],
                             model.get_params_mle(), params, len(model.module_info), dist.d, dist.valid_d, dist.N,
                             int(model.is_active.sum()), net_str])


def experiment(exp_num=1, start_id=0, data_name='celebA', dataset_path='../data/', corrupt_type='RandomPixel', gpu_id=0,
               init_delta_factor=0.0, batchsize=16, train_ite=200000, retrain_ite=500000, out_dir='./result/'):

    if gpu_id >= 0:
        torch.cuda.set_device(gpu_id)
        cudnn.benchmark = True
        cudnn.enabled = True

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Corrupt function
    if corrupt_type == 'RandomPixel':
        corrupt_func = RandomPixelMasking()
    elif corrupt_type == 'RandomHalf':
        corrupt_func = RandomHalfMasking()
    elif corrupt_type == 'Center':
        corrupt_func = CenterMasking()
    else:
        print('Invalid corrupt function type!')
        return

    train_res = SaveResult(res_file_name=out_dir + 'train_result.csv')
    retrain_res = SaveResult(res_file_name=out_dir + 'retrain_result.csv')
    with open(out_dir + 'description.txt', 'w') as o:
        o.write('data_name: ' + data_name + '\n')
        o.write('corrupt_func: ' + corrupt_type + '\n')
        o.write('batchsize: %d\n' % batchsize)
        o.write('train_ite: %d\n' % train_ite)
        o.write('retrain_ite: %d\n' % retrain_ite)

    train_data, test_data = load_data(path=dataset_path, data_name=data_name, img_size=64)
    ch_size = train_data[0][0].shape[0]

    for n in np.arange(start_id, start_id + exp_num):
        prefix = out_dir + '{:02d}_'.format(n)

        print('Architecture Search...')
        nn_model = ProbablisticCAE(in_ch_size=ch_size, out_ch_size=ch_size, row_size=1, col_size=20, level_back=5,
                                   downsample=True, k_sizes=(1, 3, 5), ch_nums=(64, 128, 256), skip=(True, False),
                                   M=None, delta_init_factor=init_delta_factor)
        optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.025, momentum=0.9, weight_decay=0., nesterov=False)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_ite)

        # Training
        period = {'max_ite': train_ite, 'save': train_ite/50, 'verbose_ite': 100}
        n_model, train_time = \
            arch_search_valid(nn_model, train_data, test_data, corrupt_func, optimizer, lr_scheduler, clip_value=5.,
                              batchsize=batchsize, lam=2, valid_rate=0.5, gpu_id=gpu_id, period=period,
                              out_model=prefix + 'trained_model.pt', log_file=prefix + 'train_log.csv')

        # Testing
        res = evaluate(nn_model, test_data, corrupt_func, gpu_id=gpu_id, batchsize=batchsize,
                       img_out_dir=prefix+'trained_model_out_img/')

        train_res.save(n, nn_model, train_time, res)  # Save result

        # Load theta from log file
        #import pandas as pd
        #df = pd.read_csv(prefix + 'train_log.csv')
        #theta = np.array(df.iloc[-1, 14:])
        #nn_model = ProbablisticCAE(in_ch_size=ch_size, out_ch_size=ch_size, row_size=1, col_size=20, level_back=5,
        #                           downsample=True, k_sizes=(1, 3, 5), ch_nums=(64, 128, 256), skip=(True, False),
        #                           M=None)
        #nn_model.asng.load_theta_from_log(theta)

        print('Retraining...')
        nn_model = ProbablisticCAE(in_ch_size=ch_size, out_ch_size=ch_size, row_size=1, col_size=20, level_back=5,
                                   downsample=True, k_sizes=(1, 3, 5), ch_nums=(64, 128, 256), skip=(True, False),
                                   M=nn_model.asng.mle())
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001, betas=(0.9, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(retrain_ite*2/5),
                                                                                   int(retrain_ite*4/5)], gamma=0.1)

        # Re-training
        period = {'max_ite': retrain_ite, 'save': retrain_ite/50, 'verbose_ite': 100}
        nn_model, train_time = train(nn_model, train_data, test_data, corrupt_func, optimizer, lr_scheduler,
                                     clip_value=5., batchsize=batchsize, gpu_id=gpu_id, period=period,
                                     out_model=prefix + 'retrained_model.pt', log_file=prefix + 'retrain_log.csv')

        # Testing
        res = evaluate(nn_model, test_data, corrupt_func, gpu_id=gpu_id, batchsize=batchsize,
                       img_out_dir=prefix + 'retrained_model_out_img/')

        retrain_res.save(n, nn_model, train_time, res)  # Save result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ASNG-NAS (Cat) for Inpainting')
    parser.add_argument('--exp_id_start', '-s', type=int, default=0, help='Starting index number of experiment')
    parser.add_argument('--exp_num', '-e', type=int, default=1, help='Number of experiments')
    parser.add_argument('--data_path', '-p', default='../data/', help='Data path')
    parser.add_argument('--data_name', '-d', default='celebA', help='Data name (celebA / cars / svhn)')
    parser.add_argument('--corrupt_type', '-c', default='RandomPixel',
                        help='Corrupt function (RandomPixel / RandomHalf / Center)')
    parser.add_argument('--gpu_id', '-g', type=int, default=0, help='GPU ID')

    parser.add_argument('--init_delta_factor', '-f', type=float, default=0.0, help='Init delta factor')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='Mini-batch size')
    parser.add_argument('--train_ite', '-t', type=int, default=50000,
                        help='Maximum number of training iterations (W updates)')
    parser.add_argument('--retrain_ite', '-r', type=int, default=500000,
                        help='Maximum number of re-training iterations (W updates)')
    parser.add_argument('--out_dir', '-o', default='./result/', help='Output directory')
    args = parser.parse_args()

    start_id = args.exp_id_start
    exp_num = args.exp_num
    data_path = args.data_path
    data_name = args.data_name
    corrupt_type = args.corrupt_type
    gpu_id = args.gpu_id
    init_delta_factor = args.init_delta_factor
    batch_size = args.batch_size
    train_ite = args.train_ite
    retrain_ite = args.retrain_ite
    out_dir = args.out_dir + data_name + '_' + corrupt_type + '/'

    experiment(exp_num=exp_num, start_id=start_id, data_name=data_name, dataset_path=data_path,
               corrupt_type=corrupt_type, gpu_id=gpu_id, init_delta_factor=init_delta_factor, batchsize=batch_size,
               train_ite=train_ite, retrain_ite=retrain_ite, out_dir=out_dir)
