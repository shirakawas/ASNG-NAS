#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import itertools
import numpy as np
from torch import nn

from .asng_cat import AdaptiveSNG
import common.operations as op


class ProbablisticCAE(nn.Module):
    in_num = 1
    out_num = 1

    def __init__(self, in_ch_size=3, out_ch_size=3, row_size=1, col_size=20, level_back=5, downsample=True,
                 k_sizes=(1, 3, 5), ch_nums=(64, 128, 256), skip=(True, False), M=None, delta_init_factor=0.):
        super(ProbablisticCAE, self).__init__()
        self.in_ch_size = in_ch_size
        self.out_ch_size = out_ch_size
        self.cols = col_size
        self.rows = row_size
        self.level_back = level_back
        self.downsample = downsample
        self.max_args = 1

        self.k_sizes = list(k_sizes)
        self.skip = list(skip)
        self.ch_nums = list(ch_nums)
        self.max_ch = np.max(self.ch_nums)

        self.module_num = len(self.k_sizes) * len(self.skip)
        self.module_params = list(itertools.product(self.k_sizes, self.skip))
        self.is_active = np.empty(self.rows * self.cols + self.out_num)

        # Categorical distribution
        categories = self.get_categories()
        N = np.sum(categories - 1)
        self.asng = AdaptiveSNG(categories, delta_init=1 / (N**delta_init_factor), lam=2)

        # list of modules for consisting hidden and output nodes: list of tuple (id, name str, arg num, is skip)
        self.module_info = []
        j = 0
        # Convolution part
        for i in range(self.cols * self.rows):
            module = []
            for m in range(self.module_num):
                k_size, skip = self.module_params[m]
                module.append((j, 'Conv_' + str(i) + '_{}'.format(self.module_params[m]), 1, skip))
                j += 1
            self.module_info.append(module)
        # Output layer
        self.module_info.append([(j, 'OutDeconv', 1, None)])

        # Create Chains
        if M is None:
            self.init_network()
        else:
            self.init_network_by_M(M)

        print('Num. of nodes (conv + out): {}'.format(len(self.module_info)))
        print('Num. of dimension of the categorical distribution: {}'.format(self.asng.d))
        print('Num. of valid dimension of the categorical distribution: {}'.format(self.asng.valid_d))
        print('Num. of params in the categorical distribution: {}'.format(self.asng.N))
        print('Num. of learnable weight parameters: {}'.format(np.sum(np.prod(param.size()) for param in self.parameters())))

    def forward_mle(self, x):
        h, M = self.forward(x, stochastic=False)
        return h

    def forward(self, x, stochastic=True):
        M = self.asng.sampling() if stochastic else self.asng.mle()
        h = self.forward_as(M, x)
        return h, M

    def forward_as(self, M, x):
        net_info = self.gene_to_net_info(M.argmax(axis=1))
        self.check_active(net_info)    # Check active modules

        f = self.f
        h = x
        active_list = []
        h_stack = []

        # Convolution part
        for i in range(self.rows * self.cols):
            if self.is_active[i]:
                k, _, _, is_skip = self.module_info[i][net_info[i][0]]
                ch_num = self.ch_nums[net_info[i][1]]
                # skip processing because the image is not down sampled in this case
                if self.downsample and not is_skip and (h.shape[2] == 1 or h.shape[3] == 1):
                    continue
                active_list.append(i)
                h = f[k](h, ch_num=ch_num)
                if is_skip:
                    h_stack.append(h)

        # Deconvolution part
        conv_num = self.rows * self.cols * self.module_num
        for i in active_list[::-1]:
            k, _, _, is_skip = self.module_info[i][net_info[i][0]]
            ch_num = self.ch_nums[net_info[i][1]]
            if is_skip:
                xx = h_stack.pop()
                h = f[k + conv_num](h, xx, ch_num=ch_num)
            else:
                h = f[k + conv_num](h, ch_num=ch_num)

        # Output
        h = f[-1](h)
        return h

    def get_categories(self):
        # list of numbers of categories
        categories = []
        # Convolution part
        for i in range(self.cols * self.rows):
            # number of modules
            categories += [self.module_num]
            # numver of channels
            categories += [len(self.ch_nums)]
            # input candidate num
            c = i // self.rows
            k = self.level_back * self.rows if c - self.level_back >= 0 else c * self.rows + self.in_num
            categories += [k] * self.max_args

        # Output layer
        categories += [1] + [1]
        k = self.level_back * self.rows if self.cols - self.level_back >= 0 else self.cols * self.rows + self.in_num
        categories += [k]
        return np.array(categories)

    def init_network(self):
        self.f = nn.ModuleList([])
        # Convolution part
        for i in range(self.cols * self.rows):
            for m in range(self.module_num):
                k_size, skip = self.module_params[m]
                stride = 2 if self.downsample and not skip else 1
                self.f.append(op.ConvUnit(in_ch=self.max_ch, k_size=k_size, pad_size=k_size // 2, stride=stride,
                                          out_ch=self.max_ch))
        # Deconvolution part
        for i in range(self.cols * self.rows):
            for m in range(self.module_num):
                k_size, skip = self.module_params[m]
                stride = 2 if self.downsample and not skip else 1
                self.f.append(op.DeconvUnit(in_ch=self.max_ch, k_size=k_size, pad_size=k_size // 2, stride=stride,
                                            out_ch=self.max_ch))
        # Output layer
        self.f.append(op.OutputDeconv(in_ch=self.max_ch, k_size=3, pad_size=1, stride=1, out_ch=self.out_ch_size))

    def init_network_by_M(self, M):
        self.asng.theta = M
        net_info = self.gene_to_net_info(M.argmax(axis=1))
        self.check_active(net_info)  # Check active modules

        self.f = nn.ModuleList([])
        in_list = [self.in_ch_size]
        # Convolution part
        for i in range(self.cols * self.rows):
            for m in range(self.module_num):
                k_size, skip = self.module_params[m]
                stride = 2 if self.downsample and not skip else 1
                if self.is_active[i] and net_info[i][0] == m:
                    ch_num = self.ch_nums[net_info[i][1]]
                    self.f.append(op.ConvUnit(in_ch=in_list[-1], k_size=k_size, pad_size=k_size // 2, stride=stride,
                                              out_ch=ch_num))
                    in_list.append(ch_num)
                else:
                    self.f.append(None)
        # Deconvolution part
        j = 2
        for i in range(self.cols * self.rows):
            for m in range(self.module_num):
                k_size, skip = self.module_params[m]
                stride = 2 if self.downsample and not skip else 1
                if self.is_active[i] and net_info[i][0] == m:
                    ch_num = self.ch_nums[net_info[i][1]]
                    self.f.append(op.DeconvUnit(in_ch=in_list[j], k_size=k_size, pad_size=k_size // 2, stride=stride,
                                                out_ch=ch_num))
                    j = np.minimum(j + 1, len(in_list) - 1)
                else:
                    self.f.append(None)

        # Output layer
        self.f.append(op.OutputDeconv(in_ch=in_list[1], k_size=3, pad_size=1, stride=1, out_ch=self.out_ch_size))

    def __check_course_to_out(self, net_info, n):
        # n is range of (0 <= n < len(self.links) - self.in_num), i.e., node No. of hidden and output
        if not self.is_active[n]:
            self.is_active[n] = True
            for in_node in net_info[n][2:]:
                if in_node >= self.in_num:
                    self.__check_course_to_out(net_info, in_node - self.in_num)

    def check_active(self, net_info):
        self.is_active[:] = False  # clear
        for i in range(self.out_num):
            self.__check_course_to_out(net_info, len(self.module_info) - i - 1)  # start from outputs

    def gene_to_net_info(self, gene):
        # list of [module type (int), ch num (int), arg 1, arg 2...] for hidden and output nodes
        net_info = []
        p = 0
        for i, m in enumerate(self.module_info):
            c = i // self.rows
            min_index = (c - self.level_back) * self.rows + self.in_num if c - self.level_back >= 0 else 0
            _, _, args, _ = m[gene[p]]
            net_info.append([gene[p]] + [gene[p + 1]] + [min_index + gene[p + j + 2] for j in range(args)])
            p += self.max_args + 2
        return net_info

    def get_params(self, net_info):
        self.check_active(net_info)  # Check active modules
        conv_num = self.rows * self.cols * self.module_num
        param_num = 0
        f = self.f
        for i in range(self.rows * self.cols):
            if self.is_active[i]:
                j, _, _, _ = self.module_info[i][net_info[i][0]]
                # conv part and output
                param_num += np.sum(np.prod(param.size()) for param in f[j].parameters())
                # deconv part
                param_num += np.sum(np.prod(param.size()) for param in f[j + conv_num].parameters())
        # output layer
        param_num += np.sum(np.prod(param.size()) for param in f[-1].parameters())
        return param_num

    def get_params_mle(self):
        M_one = self.asng.mle()
        net_info = self.gene_to_net_info(M_one.argmax(axis=1))
        return self.get_params(net_info)

    def network_string(self, net_info, sep='\n'):
        self.check_active(net_info)  # Check active modules
        net_str = ''
        for i, m in enumerate(self.module_info):
            if self.is_active[i]:
                _, name, args, _ = m[net_info[i][0]]
                if i == len(self.module_info) - 1:  # Output node
                    ch_num = self.out_ch_size
                else:
                    ch_num = self.ch_nums[net_info[i][1]]
                for j in range(args):
                    in_n = net_info[i][j+2]
                    if in_n >= self.in_num:
                        in_module_id = net_info[in_n - self.in_num][0]
                        in_ch_num = self.ch_nums[net_info[in_n - self.in_num][1]]
                        net_str += self.module_info[in_n - self.in_num][in_module_id][1] + '_' + str(in_ch_num) + ' -> '
                        net_str += name + '_' + str(ch_num) + ';' + sep
                    else:
                        net_str += 'Input_%d' % in_n + ' -> ' + name + '_' + str(ch_num) + ';' + sep
        return net_str

    def mle_network_string(self, sep='\n'):
        M_one = self.asng.mle()
        net_info = self.gene_to_net_info(M_one.argmax(axis=1))
        return self.network_string(net_info, sep=sep)

    def p_model_update(self, M_one, losses, range_restriction=True):
        self.asng.update(M_one, losses, range_restriction)
