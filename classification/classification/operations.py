#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OpsServer:
    """"Abstract class of server of available operations.

    Define search space of operations by inheriting this class.
    get_ops() must be overrided.
    """
    def __init__(self):
        self.n_ops = len(self.get_ops(1))
    
    @abstractmethod
    def get_ops(self, in_out_channels, reduction=False):
        """Method which returns a list of operation instances.

        Args:
            in_out_channels (int): The number of channels of input and output.
            reduction (bool): If true, this method must return reduction operations.

        Returns:
            list: It must contains instances of classes in torch.nn.
        """
        raise NotImplementedError


class ENASOpsServer(OpsServer):
    """OpsServer of ENAS's micro search space."""

    def __init__(self, affine=False):
        self.affine = affine
        super(ENASOpsServer, self).__init__()

    def get_ops(self, in_out_channels, reduction=False):
        """
        Args:
            in_out_channels (int): The number of channels of input and output.
            reductoin (bool): If True, the stride of operations is set to 2.
        Returns:
            list: 5 operations are contained: identity, 3x3 average pooling, 3x3 max pooling,
                and 3x3 and 5x5 separable convolutions.
        """
        stride = 2 if reduction else 1
        ops = [
            Identity() if not reduction else FactorizedReduce(in_out_channels, in_out_channels, affine=self.affine),
            nn.AvgPool2d(3, stride=stride, padding=1),
            nn.MaxPool2d(3, stride=stride, padding=1),
            SepConv(in_out_channels, in_out_channels, 3, stride, padding=1, affine=self.affine),
            SepConv(in_out_channels, in_out_channels, 5, stride, padding=2, affine=self.affine)
        ]
        return ops


class FactorizedReduce(nn.Module):
    """Reduction operation preserving information of features."""
    def __init__(self, in_channels, out_channels, affine=True):
        super(FactorizedReduce, self).__init__()
        assert out_channels % 2 == 0
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(in_channels, out_channels//2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(in_channels, out_channels//2, 1, stride=2, padding=0, bias=False) 
        self.bn = nn.BatchNorm2d(out_channels, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        h = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        h = self.bn(h)
        return h


class Identity(nn.Module):
    """Identity operation."""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ReLUConvBN(nn.Module):
    """
    -> ReLU
    -> kernel_size convolution
    -> Batch Normalization
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """Depthwise Separable Convolution x2.
    
    Apply this operations twice.
    -> ReLU
    -> kernel_size depthwise convolution
    -> 1x1 convolution
    -> Batch Normalization
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels, affine=affine),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class AdaptiveConv(nn.Module):
    """Adaptive Convolution.

    The number of input's channels vary following given indices.

    Args:
        max_in_channels (int): The maximum number of input's channels.
        out_channels (int): The number of output's channels.
        kernel_size (int or tuple): Size of the kernel.
        stride (int): Stride of the convolution.
        padding (int): Width of zero-padding.
        bias (bool): If True, learnable bias will be added.
    """
    def __init__(self, max_in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(AdaptiveConv, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.fixed = False
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W  = nn.Parameter(torch.Tensor(out_channels, max_in_channels, *kernel_size))
        self.b = None
        if bias:
            self.b = nn.Parameter(torch.Tensor(out_channels))

        nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))
        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.b, -bound, bound)



    def forward(self, X, ind):
        """
        Args:
            X (torch.FloatTensor): Input.
            ind (torch.LongTensor): Indices of channel.
                The number of indices and input's channels must be same.
        """
        if not self.fixed:
            assert ind is not None
            assert X.size(1) == len(ind)

            W = torch.index_select(self.W, dim=1, index=ind)
            return F.conv2d(X, W, self.b, self.stride, self.padding)
        else:
            assert X.size(1) == self.W.size(1)
            return F.conv2d(X, self.W, self.b, self.stride, self.padding)


    def fix_arc(self, ind):
        self.W = nn.Parameter(torch.index_select(self.W, dim=1, index=ind))
        self.fixed = True


    def get_params(self, ind):
        if not self.fixed:
            W = torch.index_select(self.W, dim=1, index=ind)
            n_params = np.prod(W.size())
        else:
            n_params = np.prod(self.W.size())

        if self.b is not None:
            n_params += self.b.size()[0]

        return n_params
