#!/usr/bin/env python
# -*- coding: utf-8 -*-

import classification.operations as O
import numpy as np
import torch
from classification import utils
from torch import nn

from util.categorical_asng import CategoricalASNG

n_prev_cells = 2
n_categorical_dist_per_node = 4
channels_image = 3


def count_params(module):
    return np.sum(np.prod(param.size()) for param in module.parameters())


class Cell(nn.Module):
    """Cell used in ENAS's micro search space.

    Args:
        ops_server (operations.OpsServer): Server of a list of operation instances.
        in_channels1 (int): The number of channels of input from the cell before the previous cell.
        in_channels2 (int): The number of channels of input from the previous cell.
        out_channels (int): The number of channels of output.
        n_nodes (int): The number of nodes in a cell.
        reduction (bool): If True, the stride of the operations adjacent to inputs become 2.
        after_reduction (bool): Make it True iff the previous cell is a Reduction Cell.
            If True, inputs from the cell before the privious cell will be modified by reduction operation.
        affine (bool): If True, BatchNormalization has learnable affine parameters.
    """

    def __init__(self, ops_server, in_channels1, in_channels2, out_channels, n_nodes, reduction, after_reduction, affine):
        super(Cell, self).__init__()

        self.n_nodes = n_nodes
        self.out_channels = out_channels
        self.n_ops = ops_server.n_ops

        if after_reduction:
            self.preprocess1 = O.FactorizedReduce(in_channels1, out_channels, affine=affine)
        else:
            self.preprocess1 = O.ReLUConvBN(in_channels1, out_channels, kernel_size=1, stride=1, padding=0, affine=affine)
        self.preprocess2 =  O.ReLUConvBN(in_channels2, out_channels, kernel_size=1, stride=1, padding=0, affine=affine)

        self.op = nn.ModuleList([])
        for node in range(n_nodes):
            op_row = nn.ModuleList([])
            for prev_node in range(node + n_prev_cells):
                # Make operations' stride 2
                # iff this object is Reduction Cell and operations are adjacent to inputs from previous cells.
                ops = ops_server.get_ops(out_channels, reduction=(reduction and prev_node < n_prev_cells))
                ops = nn.ModuleList(ops)
                op_row.append(ops)
            self.op.append(op_row)

        self.relu = nn.ReLU()
        self.conv = O.AdaptiveConv(n_nodes*out_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, affine=affine)

    def forward(self, M, h1, h2, drop_path_rate=0.):
        """
        Args:
            M (numpy.ndarray): One-hot vectors representing a model.
                Its shape must be (n_categorical_dist_per_node * n_nodes, max_n_categories).
            h1 (Tensor): The output of the cell before the previous cell.
            h2 (Tensor): The output of the previous cell.
        """

        assert M.ndim == 2

        # Decode one-hot vectors to intergers (0-indexed).
        M_dec = M.argmax(axis=1)

        # Forward propagation.
        h1 = self.preprocess1(h1)
        h2 = self.preprocess2(h2)

        hs = [h1, h2]
        used = np.zeros(n_prev_cells + self.n_nodes, dtype=bool)
        for i in range(self.n_nodes):
            ind = i*n_categorical_dist_per_node

            inp1 = hs[M_dec[ind]]
            inp2 = hs[M_dec[ind + 1]]
            op1 = self.op[i][M_dec[ind]][M_dec[ind + 2]]
            op2 = self.op[i][M_dec[ind + 1]][M_dec[ind + 3]]
            h1 = op1(inp1)
            h2 = op2(inp2)

            if not isinstance(op1, O.Identity):
                h1 = utils.drop_path(h1, drop_path_rate)
            if not isinstance(op2, O.Identity):
                h2 = utils.drop_path(h2, drop_path_rate)

            h = h1 + h2
            hs.append(h)

            used[M_dec[ind]] = used[M_dec[ind + 1]] = True

        # Concatenate unused outputs.
        used[0] = used[1] = True
        ind = np.arange(self.n_nodes, dtype=int)[np.logical_not(used[-self.n_nodes:])]
        ind = [self.out_channels*i + j for i in ind for j in range(self.out_channels)]
        ind = torch.LongTensor(ind)
        if self.conv.W.is_cuda:
            ind = ind.cuda(self.conv.W.get_device())
        h = torch.cat([hs[i] for i in range(len(used)) if not used[i]], dim=1)

        # 1x1 convolution to modify output's shape.
        h = self.relu(h)
        h = self.conv(h, ind)
        h = self.bn(h)

        return h

    def get_params(self, M):
        """
        Args:
            M (numpy.ndarray): One-hot vectors representing a model.
                Its shape must be (n_categorical_dist_per_node * n_nodes, max_n_categories).
        Returns:
            int: The number of learnable parameters given model parameters M.
        """

        # Decode one-hot vectors to intergers (0-indexed).
        M_dec = M.argmax(axis=1)

        n_params = count_params(self.preprocess1) + count_params(self.preprocess2)

        used = np.zeros(n_prev_cells + self.n_nodes, dtype=bool)
        for i in range(self.n_nodes):
            ind = i*n_categorical_dist_per_node

            n_params += count_params(self.op[i][M_dec[ind]][M_dec[ind + 2]])
            if (M_dec[ind], M_dec[ind + 2]) != (M_dec[ind + 1], M_dec[ind + 3]):
                n_params += count_params(self.op[i][M_dec[ind + 1]][M_dec[ind + 3]])

            used[M_dec[ind]] = used[M_dec[ind + 1]] = True

        ind = np.arange(self.n_nodes, dtype=int)[np.logical_not(used[-self.n_nodes:])]
        ind = [self.out_channels*i + j for i in ind for j in range(self.out_channels)]
        ind = torch.LongTensor(ind)
        if self.conv.W.is_cuda:
            ind = ind.cuda(self.conv.W.get_device())
        n_params += self.conv.get_params(ind)
        n_params += count_params(self.bn)

        return n_params

    def fix_arc(self, M):
        M_dec = M.argmax(axis=1)
        used = []
        for i in range(self.n_nodes):
            ind = i*n_categorical_dist_per_node
            used.append((i, M_dec[ind], M_dec[ind + 2]))
            used.append((i, M_dec[ind + 1], M_dec[ind + 3]))

        for i in range(self.n_nodes):
            for j in range(i + n_prev_cells):
                for k in range(self.n_ops):
                    if (i, j, k) not in used:
                        self.op[i][j][k] = None

        # Fix adaptive conv.
        used = np.zeros(n_prev_cells + self.n_nodes, dtype=bool)
        for i in range(self.n_nodes):
            ind = i*n_categorical_dist_per_node
            used[M_dec[ind]] = used[M_dec[ind + 1]] = True

        used[0] = used[1] = True
        ind = np.arange(self.n_nodes, dtype=int)[np.logical_not(used[-self.n_nodes:])]
        ind = [self.out_channels*i + j for i in ind for j in range(self.out_channels)]
        ind = torch.LongTensor(ind)
        if self.conv.W.is_cuda:
            ind = ind.cuda(self.conv.W.get_device())
        self.conv.fix_arc(ind)


class MicroProbabilisticCNN(nn.Module):
    """Probabilistic CNN with ENAS's micro search space.

    Args:
        n_classes (int): The number of classes. It is equal to the dimension of output layer.
        channels_init (int): The number of channels in the first Cell.
        n_cells (int): The number of cells. It contains two Reduction Cells.
        n_nodes (int): The number of nodes in each cell.
        stem_multiplier (int): The number of channels of input firstly increase to
            channels_init*stem_multiplier by applying 3x3 Convolution.
        affine (bool): If True, BatchNormalization has learnable affine parameters.
        ops_server (operations.OpsServer): Server of a list of operation instances.
        alpha (float): Threshold of SNR in ASNG algorithm.
        init_delta (float): Initial value of delta in ASNG algorithm.
        trained_theta (numpy.ndarray): Pre-trained theta of categorical distributions.
            Its shape must be (n_categorical_dist_per_node * n_nodes * n_cells * 2, max_n_categories).
            If it is None, theta is initialized by 1/n_categories.
    """

    def __init__(self, n_classes, channels_init=16, n_cells=8, n_nodes=5, stem_multiplier=3,
                 affine=False, ops_server=None, alpha=1.5, init_delta=1.0, trained_theta=None):
        super(MicroProbabilisticCNN, self).__init__()

        self.n_classes = n_classes
        self.n_cells = n_cells
        self.n_nodes = n_nodes
        if ops_server is None:
            ops_server = O.ENASOpsServer()
        if ops_server.__class__.__name__ != 'ENASOpsServer':
            print('***CAUTION*** ops_server is not ENASOpsServer but %s' % ops_server.__class__.__name__)

        channels_cur = channels_init*stem_multiplier
        self.stem = nn.Sequential(
            nn.Conv2d(channels_image, channels_cur, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels_cur)
        )

        # channels_prev1 is the number of channels of the cell before the previous cell.
        # channels_prev2 is the number of channels of the previous cell.
        channels_prev1, channels_prev2, channels_cur = channels_cur, channels_cur, channels_init
        after_reduction = False
        channels_auxiliary = None

        self.cell = nn.ModuleList([])
        for i in range(n_cells):
            if i in (n_cells//3, 2*n_cells//3):
                channels_cur *= 2
                reduction = True
            else:
                reduction = False
            self.cell.append(Cell(ops_server, channels_prev1, channels_prev2, channels_cur, n_nodes, reduction, after_reduction, affine))
            after_reduction = reduction
            channels_prev1, channels_prev2 = channels_prev2, channels_cur
            if i == 2*n_cells//3:
                channels_auxiliary = channels_prev2

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels_prev2, self.n_classes)

        # Definition of auxiliary network.
        self.auxiliary_network = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(channels_auxiliary, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 768, kernel_size=2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.auxiliary_classifier = nn.Linear(768, n_classes)

        # Initialize categorical distribution.
        self.n_ops = ops_server.n_ops
        self.categories = []
        for n_prev_nodes in range(n_nodes):
            self.categories += [n_prev_cells + n_prev_nodes, n_prev_cells + n_prev_nodes, self.n_ops, self.n_ops]
        self.categories *= 2
        self.categories = np.array(self.categories)
        self.asng = CategoricalASNG(self.categories, alpha=alpha, init_delta=init_delta, init_theta=trained_theta)

    @property
    def p_model(self):
        return self.asng.p_model

    def forward(self, X, stochastic, drop_path_rate=0.):
        """
        Args:
            X (Tensor): Inputs.
            stochastic (bool): If True, use random architecture drawn from categorical distribution.
                Otherwise, use the most likely architecture.
            drop_path_rate (float): Dropout rate of Droppath.
        """
        M = self.asng.sampling() if stochastic else self.p_model.mle()
        M_normal = M[:len(M)//2]
        M_reduction = M[len(M)//2:]

        h1 = h2 = self.stem(X)
        h_auxiliary = None
        for i in range(self.n_cells):
            if i in (self.n_cells//3, 2*self.n_cells//3):
                h1, h2 = h2, self.cell[i](M_reduction, h1, h2, drop_path_rate)
            else:
                h1, h2 = h2, self.cell[i](M_normal, h1, h2, drop_path_rate)

            if i == 2*self.n_cells//3:
                h_auxiliary = self.auxiliary_network(h2)
                h_auxiliary = self.auxiliary_classifier(h_auxiliary.view(h_auxiliary.size(0), -1))

        h = self.global_pooling(h2)
        h = self.classifier(h.view(h.size(0), -1))

        return h, M, h_auxiliary

    def forward_as(self, M, X):
        """Forward with specified architecture.

        Args:
            M (numpy.ndarray): Model parameter.
            X (Tensor): Inputs.
        """

        assert M.shape == self.p_model.theta.shape

        M_normal = M[:len(M)//2]
        M_reduction = M[len(M)//2:]

        h1 = h2 = self.stem(X)
        for i in range(self.n_cells):
            if i in (self.n_cells//3, 2*self.n_cells//3):
                h1, h2 = h2, self.cell[i](M_reduction, h1, h2)
            else:
                h1, h2 = h2, self.cell[i](M_normal, h1, h2)

        h = self.global_pooling(h2)
        h = self.classifier(h.view(h.size(0), -1))

        return h

    def p_model_update(self, M, losses, range_restriction=True):
        self.asng.update(M, losses, range_restriction)

    def get_params(self, M):
        """
        Args:
            M (numpy.ndarray): One-hot vectors representing a model.
                Its shape must be (n_categorical_dist_per_node * n_nodes * n_cells * 2, max_n_categories).
        Returns
            int: The number of learnable parameters given model parameters M.
                It doesn't include parameters of auxiliary network.
        """
        M_normal = M[:len(M)//2]
        M_reduction = M[len(M)//2:]

        n_params = count_params(self.stem)
        for i in range(self.n_cells):
            if i in (self.n_cells//3, 2*self.n_cells//3):
                n_params += self.cell[i].get_params(M_reduction)
            else:
                n_params += self.cell[i].get_params(M_normal)
        n_params += count_params(self.classifier)

        return n_params

    def get_params_mle(self):
        """
        Returns:
            int: The number of learnable parameters given the most likely model paramters.
        """
        M = self.p_model.mle()
        return self.get_params(M)

    def fix_arc(self):
        """Remove all operations except for those used in the most likely model."""
        M = self.p_model.mle()
        M_normal = M[:len(M)//2]
        M_reduction = M[len(M)//2:]
        for i in range(self.n_cells):
            if i in (self.n_cells//3, 2*self.n_cells//3):
                self.cell[i].fix_arc(M_reduction)
            else:
                self.cell[i].fix_arc(M_normal)
