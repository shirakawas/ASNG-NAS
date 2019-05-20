#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class AdaptiveSNG:
    """
    Adaptive Stochastic Natural Gradient for Categorical Distribution
    """
    def __init__(self, categories, alpha=1.5, delta_init=1., lam=2, Delta_max=np.inf, init_theta=None):

        self.N = np.sum(categories - 1)

        # Categorical distribution
        self.d = len(categories)
        self.C = categories
        self.Cmax = np.max(categories)
        self.theta = np.zeros((self.d, self.Cmax))
        # initialize theta by 1/C for each dimensions
        for i in range(self.d):
            self.theta[i, :self.C[i]] = 1. / self.C[i]
        # pad zeros to unused elements
        for i in range(self.d):
            self.theta[i, self.C[i]:] = 0.
        # valid dimension size
        self.valid_d = len(self.C[self.C > 1])

        if init_theta is not None:
            self.theta = init_theta

        # Adaptive SG
        self.alpha = alpha  # threshold for adaptation
        self.delta_init = delta_init
        self.lam = lam  # lambda_theta
        self.Delta_max = Delta_max  # maximum Delta (can be np.inf)

        self.Delta = 1.
        self.gamma = 0.0  # correction factor
        self.s = np.zeros(self.N)  # averaged stochastic natural gradient
        self.delta = self.delta_init / self.Delta
        self.eps = self.delta

    def get_lam(self):
        return self.lam

    def get_delta(self):
        return self.delta

    def sampling(self):
        """
        Draw a sample from the categorical distribution (one-hot)
        """
        rand = np.random.rand(self.d, 1)  # range of random number is [0, 1)
        cum_theta = self.theta.cumsum(axis=1)  # (d, Cmax)

        # x[i, j] becomes 1 if cum_theta[i, j] - theta[i, j] <= rand[i] < cum_theta[i, j]
        c = (cum_theta - self.theta <= rand) & (rand < cum_theta)
        return c

    def mle(self):
        """
        Get most likely categorical variables (one-hot)
        """
        m = self.theta.argmax(axis=1)
        x = np.zeros((self.d, self.Cmax))
        for i, c in enumerate(m):
            x[i, c] = 1
        return x

    def update(self, c_one, fxc, range_restriction=True):
        self.delta = self.delta_init / self.Delta
        beta = self.delta / (self.N ** 0.5)

        aru, idx = self.utility(fxc)
        if np.all(aru == 0):
            # If all the points have the same f-value,
            # nothing happens for theta and breaks.
            # In this case, we skip the rest of the code.
            return

        ng = np.mean(aru[:, np.newaxis, np.newaxis] * (c_one[idx] - self.theta), axis=0)

        sl = []
        for i, K in enumerate(self.C):
            theta_i = self.theta[i, :K - 1]
            theta_K = self.theta[i, K - 1]
            s_i = 1. / np.sqrt(theta_i) * ng[i, :K - 1]
            s_i += np.sqrt(theta_i) * ng[i, :K - 1].sum() / (theta_K + np.sqrt(theta_K))
            sl += list(s_i)
        sl = np.array(sl)

        pnorm = np.sqrt(np.dot(sl, sl)) + 1e-9
        self.eps = self.delta / pnorm
        self.theta += self.eps * ng

        self.s = (1 - beta) * self.s + np.sqrt(beta * (2 - beta)) * sl / pnorm
        self.gamma = (1 - beta)**2 * self.gamma + beta*(2 - beta)
        self.Delta *= np.exp(beta * (self.gamma - np.dot(self.s, self.s) / self.alpha))
        self.Delta = min(self.Delta, self.Delta_max)

        for i in range(self.d):
            ci = self.C[i]
            # Constraint for theta (minimum value of theta and sum of theta = 1.0)
            theta_min = 1. / (self.valid_d * (ci - 1)) if range_restriction and ci > 1 else 0.
            self.theta[i, :ci] = np.maximum(self.theta[i, :ci], theta_min)
            theta_sum = self.theta[i, :ci].sum()
            tmp = theta_sum - theta_min * ci
            self.theta[i, :ci] -= (theta_sum - 1.) * (self.theta[i, :ci] - theta_min) / tmp
            # Ensure the summation to 1
            self.theta[i, :ci] /= self.theta[i, :ci].sum()

    @staticmethod
    def utility(f, rho=0.25, negative=True):
        """
        Ranking Based Utility Transformation

        w(f(x)) / lambda =
            1/mu  if rank(x) <= mu
            0     if mu < rank(x) < lambda - mu
            -1/mu if lambda - mu <= rank(x)

        where rank(x) is the number of at least equally good
        points, including it self.

        The number of good and bad points, mu, is ceil(lambda/4).
        That is,
            mu = 1 if lambda = 2
            mu = 1 if lambda = 4
            mu = 2 if lambda = 6, etc.

        If there exist tie points, the utility values are
        equally distributed for these points.
        """
        eps = 1e-14
        idx = np.argsort(f)
        lam = len(f)
        mu = int(np.ceil(lam * rho))
        _w = np.zeros(lam)
        _w[:mu] = 1 / mu
        _w[lam - mu:] = -1 / mu if negative else 0
        w = np.zeros(lam)
        istart = 0
        for i in range(f.shape[0] - 1):
            if f[idx[i + 1]] - f[idx[i]] < eps * f[idx[i]]:
                pass
            elif istart < i:
                w[istart:i + 1] = np.mean(_w[istart:i + 1])
                istart = i + 1
            else:
                w[i] = _w[i]
                istart = i + 1
        w[istart:] = np.mean(_w[istart:])
        return w, idx

    def log_header(self, theta_log=False):
        header_list = ['delta', 'eps', 'snorm_alpha', 'theta_converge']
        if theta_log:
            for i in range(self.d):
                header_list += ['theta%d_%d' % (i, j) for j in range(self.C[i])]
        return header_list

    def log(self, theta_log=False):
        log_list = [self.delta, self.eps, np.dot(self.s, self.s) / self.alpha, self.theta.max(axis=1).mean()]

        if theta_log:
            for i in range(self.d):
                log_list += ['%f' % self.theta[i, j] for j in range(self.C[i])]
        return log_list

    def load_theta_from_log(self, theta_log):
        self.theta = np.zeros((self.d, self.Cmax))
        k = 0
        for i in range(self.d):
            for j in range(self.C[i]):
                self.theta[i, j] = theta_log[k]
                k += 1
