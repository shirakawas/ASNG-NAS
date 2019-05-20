
import numpy as np
import torch
from torch import nn


class fxc1(nn.Module):

    def __init__(self, K=2, D=10, noise=False):
        super(fxc1, self).__init__()
        self.noise = noise
        self.factor = torch.Tensor(np.arange(0, K)) / K
        self.x = nn.Parameter(torch.randn(D, K))
        self.target = 1 / K
        self.eval = 0

    def forward(self, c):
        self.eval += 1
        # t = torch.randn(len(c), self.x.shape[0], 1) / self.x.shape[1] if self.noise else 0.  # different samples
        t = torch.randn(self.x.shape[0], 1) / self.x.shape[1] if self.noise else 0.  # same samples
        loss = torch.sum(c * (self.factor + (self.x - t)**2), dim=(1, 2))
        general_loss = torch.sum(c * (self.factor + self.x**2), dim=(1, 2))
        return loss, general_loss


class fxc2(nn.Module):

    def __init__(self, nc=10, d=10):
        super(fxc2, self).__init__()
        self.x = nn.Parameter(torch.randn(nc, d))
        self.sigmod = nn.Sigmoid()
        self.nc = nc
        self.target = 0
        self.eval = 0

    def forward(self, c):
        self.eval += 1
        fx = torch.mean(self.sigmod(self.x), 1)
        # fx = self.nc - torch.sum(c * xx, 0)  # onemax like
        fx = self.nc - torch.cumprod(c * fx, 0).sum()  # leading ones like
        fx += 1e-8 * torch.sum(self.x ** 2)
        return fx, fx
