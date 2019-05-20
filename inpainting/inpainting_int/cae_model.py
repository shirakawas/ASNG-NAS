#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
from torch import nn

from .asng_int import AdaptiveSNG
import common.operations as op


class ProbablisticCAE(nn.Module):
    in_num = 1
    out_num = 1

    def __init__(self, in_ch_size=3, out_ch_size=3, row_size=1, col_size=20, level_back=5, downsample=True,
                 k_sizes=(1, 3, 5), ch_range=(64, 256), c=None, delta_init_factor=0.):
        super(ProbablisticCAE, self).__init__()
        self.in_ch_size = in_ch_size
        self.out_ch_size = out_ch_size
        self.cols = col_size
        self.rows = row_size
        self.level_back = level_back
        self.downsample = downsample
        self.max_args = 1

        self.skip = [0, 1]
        self.k_sizes = list(k_sizes)
        self.k_range = [1, len(k_sizes)]
        self.ch_range = list(ch_range)
        self.max_ch = np.max(self.ch_range)

        self.module_num = len(self.k_sizes) * len(self.skip)
        self.is_active = np.empty(self.rows * self.cols + self.out_num)

        # Categorical distribution
        categories, integers = self.get_variable_space()
        n = np.sum(categories - 1) + 2*len(integers)
        self.asng = AdaptiveSNG(categories, integers, delta_init=1 / (n**delta_init_factor), lam=2)

        # list of modules for consisting hidden and output nodes: list of tuple (id, name str, arg num, is skip)
        self.module_info = []
        j = 0
        # Convolution part
        for i in range(self.cols * self.rows):
            module = []
            for s in self.skip:
                for k in self.k_sizes:
                    module.append((j, 'Conv_' + str(i) + '_{}_{}'.format(k, s), 1, s))
                    j += 1
            self.module_info.append(module)
        # Output layer
        self.module_info.append([(j, 'OutDeconv', 1, None)])

        # Create Chains
        if c is None:
            self.init_network()
        else:
            self.init_network_by_c(c)

        print('Num. of nodes (conv + out): {}'.format(len(self.module_info)))
        print('Num. of dimension of the categorical distribution: {}'.format(self.asng.d_cat))
        print('Num. of valid dimension of the categorical distribution: {}'.format(self.asng.valid_d_cat))
        print('Num. of params in the categorical distribution: {}'.format(self.asng.n_cat))
        print('Num. of dimension of the normal distribution: {}'.format(self.asng.d_int))
        print('Num. of params in the normal distribution: {}'.format(self.asng.n_int))
        print('Num. of weight parameters: {}'.format(np.sum(np.prod(param.size()) for param in self.parameters())))

    def forward_mle(self, x):
        c_cat, c_int = self.asng.mle()
        return self.forward(c_cat, c_int, x)

    def forward(self, c_cat, c_int, x):
        net_info = self.gene_to_net_info(c_cat.argmax(axis=1), c_int)
        self.check_active(net_info)    # Check active modules

        f = self.f
        h = x
        active_list = []
        h_stack = []

        # Convolution part
        for i in range(self.rows * self.cols):
            if self.is_active[i]:
                m_idx, _, _, is_skip = self.module_info[i][net_info[i][0]*len(self.k_sizes) + net_info[i][1]]
                ch_num = net_info[i][2]
                # skip processing because the image is not down sampled in this case
                if self.downsample and not is_skip and (h.shape[2] == 1 or h.shape[3] == 1):
                    continue
                active_list.append(i)
                h = f[m_idx](h, ch_num=ch_num)
                if is_skip:
                    h_stack.append(h)

        # Deconvolution part
        conv_num = self.rows * self.cols * self.module_num
        for i in active_list[::-1]:
            m_idx, _, _, is_skip = self.module_info[i][net_info[i][0] * len(self.k_sizes) + net_info[i][1]]
            ch_num = net_info[i][2]
            if is_skip:
                xx = h_stack.pop()
                h = f[m_idx + conv_num](h, xx, ch_num=ch_num)
            else:
                h = f[m_idx + conv_num](h, ch_num=ch_num)

        # Output
        h = f[-1](h)
        return h

    def get_variable_space(self):
        # list of numbers for categorical variables
        categories = []
        # list of ranges for integer variables
        integers = []
        # Convolution part
        for i in range(self.cols * self.rows):
            # existence of skip
            categories += [2]
            # channel and kernel size
            integers += [self.k_range]
            integers += [self.ch_range]
            # input candidate num
            c = i // self.rows
            k = self.level_back * self.rows if c - self.level_back >= 0 else c * self.rows + self.in_num
            categories += [k] * self.max_args

        # Output layer
        k = self.level_back * self.rows if self.cols - self.level_back >= 0 else self.cols * self.rows + self.in_num
        categories += [k]
        return np.array(categories), np.array(integers)

    def init_network(self):
        self.f = nn.ModuleList([])
        # Convolution part
        for i in range(self.cols * self.rows):
            for s in self.skip:
                for k in self.k_sizes:
                    stride = 2 if self.downsample and s == 0 else 1
                    self.f.append(op.ConvUnit(in_ch=self.max_ch, k_size=k, pad_size=k // 2, stride=stride,
                                              out_ch=self.max_ch))
        # Deconvolution part
        for i in range(self.cols * self.rows):
            for s in self.skip:
                for k in self.k_sizes:
                    stride = 2 if self.downsample and s == 0 else 1
                    self.f.append(op.DeconvUnit(in_ch=self.max_ch, k_size=k, pad_size=k // 2, stride=stride,
                                                out_ch=self.max_ch))
        # Output layer
        self.f.append(op.OutputDeconv(in_ch=self.max_ch, k_size=3, pad_size=1, stride=1, out_ch=self.out_ch_size))

    def init_network_by_c(self, c):
        self.asng.theta_cat = c[0]
        self.asng.theta_int[0] = c[1]
        net_info = self.gene_to_net_info(c[0].argmax(axis=1), c[1])
        self.check_active(net_info)  # Check active modules

        self.f = nn.ModuleList([])
        in_list = [self.in_ch_size]
        # Convolution part
        for i in range(self.cols * self.rows):
            for s in self.skip:
                for r, k in enumerate(self.k_sizes):
                    stride = 2 if self.downsample and s == 0 else 1
                    if self.is_active[i] and net_info[i][0] == s and net_info[i][1] == r:
                        ch_num = net_info[i][2]
                        self.f.append(op.ConvUnit(in_ch=in_list[-1], k_size=k, pad_size=k // 2, stride=stride,
                                                  out_ch=ch_num))
                        in_list.append(ch_num)
                    else:
                        self.f.append(None)
        # Deconvolution part
        j = 2
        for i in range(self.cols * self.rows):
            for s in self.skip:
                for r, k in enumerate(self.k_sizes):
                    stride = 2 if self.downsample and not s else 1
                    if self.is_active[i] and net_info[i][0] == s and net_info[i][1] == r:
                        ch_num = net_info[i][2]
                        self.f.append(op.DeconvUnit(in_ch=in_list[j], k_size=k, pad_size=k // 2, stride=stride,
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
            for in_node in net_info[n][3:]:
                if in_node >= self.in_num:
                    self.__check_course_to_out(net_info, in_node - self.in_num)

    def check_active(self, net_info):
        self.is_active[:] = False  # clear
        for i in range(self.out_num):
            self.__check_course_to_out(net_info, len(self.module_info) - i - 1)  # start from outputs

    def gene_to_net_info(self, c_cat, c_int):
        cc_int = np.round(np.clip(c_int, self.asng.int_min, self.asng.int_max))
        cc_int = cc_int.astype(np.int)
        # list of [skip (bool), k_size (int), ch_num (int), arg 1, arg 2...] for hidden and output nodes
        net_info = []
        p = q = 0
        for i, m in enumerate(self.module_info):
            c = i // self.rows
            min_index = (c - self.level_back) * self.rows + self.in_num if c - self.level_back >= 0 else 0
            if i < self.rows * self.cols:
                _, _, args, _ = m[c_cat[p]*len(self.k_sizes) + cc_int[q]-1]
                net_info.append(
                    [c_cat[p]] + [cc_int[q]-1] + [cc_int[q+1]] + [min_index + c_cat[p+j+1] for j in range(args)])
            else:  # output node
                net_info.append([0] + [0] + [0] + [min_index + c_cat[p]])

            p += self.max_args + 1
            q += 2
        return net_info

    def get_params(self, net_info):
        self.check_active(net_info)  # Check active modules
        conv_num = self.rows * self.cols * self.module_num
        param_num = 0
        f = self.f
        for i in range(self.rows * self.cols):
            if self.is_active[i]:
                j, _, _, _ = self.module_info[i][net_info[i][0] * len(self.k_sizes) + net_info[i][1]]
                # conv part and output
                param_num += np.sum(np.prod(param.size()) for param in f[j].parameters())
                # deconv part
                param_num += np.sum(np.prod(param.size()) for param in f[j + conv_num].parameters())
        # output layer
        param_num += np.sum(np.prod(param.size()) for param in f[-1].parameters())
        return param_num

    def get_params_mle(self):
        c_cat, c_int = self.asng.mle()
        net_info = self.gene_to_net_info(c_cat.argmax(axis=1), c_int)
        return self.get_params(net_info)

    def network_string(self, net_info, sep='\n'):
        # list of [skip (bool), k_size (int), ch_num (int), arg 1, arg 2...] for hidden and output nodes
        self.check_active(net_info)  # Check active modules
        net_str = ''
        for i, m in enumerate(self.module_info):
            if self.is_active[i]:
                _, name, args, _ = m[net_info[i][0] * len(self.k_sizes) + net_info[i][1]]
                if i == len(self.module_info) - 1:  # Output node
                    ch_num = self.out_ch_size
                else:
                    ch_num = net_info[i][2]
                for j in range(args):
                    in_n = net_info[i][j+3]
                    if in_n >= self.in_num:
                        in_module_id = net_info[in_n - self.in_num][0] * len(self.k_sizes) + net_info[in_n - self.in_num][1]
                        in_ch_num = net_info[in_n - self.in_num][2]
                        net_str += self.module_info[in_n - self.in_num][in_module_id][1] + '_' + str(in_ch_num) + ' -> '
                        net_str += name + '_' + str(ch_num) + ';' + sep
                    else:
                        net_str += 'Input_%d' % in_n + ' -> ' + name + '_' + str(ch_num) + ';' + sep
        return net_str

    def mle_network_string(self, sep='\n'):
        c_cat, c_int = self.asng.mle()
        net_info = self.gene_to_net_info(c_cat.argmax(axis=1), c_int)
        return self.network_string(net_info, sep=sep)

    def p_model_update(self, c, losses, range_restriction=True):
        self.asng.update(c, losses, range_restriction)
