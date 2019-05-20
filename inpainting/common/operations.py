#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F


def weight_initialization(layer):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')


def ch_pad_clip(x, trg_ch_size):
    if trg_ch_size > x.shape[1]:
        # Pad channels to x
        x = F.pad(x, (0, 0, 0, 0, 0, int(trg_ch_size - x.shape[1])), 'constant', value=0)
    elif trg_ch_size < x.shape[1]:
        # Delete channels of x
        x = x[:, :trg_ch_size, :, :]
    return x


# Conv -> ReLU
class ConvUnit(nn.Module):
    def __init__(self, in_ch=256, k_size=3, pad_size=1, stride=1, out_ch=64):
        super(ConvUnit, self).__init__()
        self.in_ch = in_ch
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, out_ch, k_size, padding=pad_size, stride=stride, bias=False)
        # weight_initialization(self.conv)

    def forward(self, x, ch_num=None):
        x = ch_pad_clip(x, self.in_ch)
        x = self.conv(x) if ch_num is None else self.conv(x)[:, :ch_num]
        return self.relu(x)


# [Deconv(x) + h]  -> ReLU or Deconv -> ReLU
class DeconvUnit(nn.Module):
    def __init__(self, in_ch=256, k_size=3, pad_size=1, stride=1, out_ch=64):
        super(DeconvUnit, self).__init__()
        self.in_ch = in_ch
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, k_size, padding=pad_size, stride=stride,
                                       output_padding=stride-1, bias=False)
        # weight_initialization(self.conv)

    def forward(self, x, h=None, ch_num=None):
        x = ch_pad_clip(x, self.in_ch)
        x = self.conv(x) if ch_num is None else self.conv(x)[:, :ch_num]
        if h is None:
            return self.relu(x)
        else:
            return self.relu(x + h)


# Deconv
class OutputDeconv(nn.Module):
    def __init__(self, in_ch=256, k_size=3, pad_size=1, stride=1, out_ch=64):
        super(OutputDeconv, self).__init__()
        self.in_ch = in_ch
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, k_size, padding=pad_size, stride=stride, bias=False)
        # weight_initialization(self.conv)

    def forward(self, x):
        x = ch_pad_clip(x, self.in_ch)
        return self.conv(x)
