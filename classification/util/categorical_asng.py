#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from util.categorical_dist import Categorical


class CategoricalASNG:
    """Adaptive stochastic natural gradient method on multivariate categorical distribution.

    Args:
        categories (numpy.ndarray): Array containing the numbers of categories of each dimension.
        alpha (float): Threshold of SNR in ASNG algorithm.
        init_delta (float): Initial value of delta.
        Delta_max (float): Maximum value of Delta.
        init_theta (numpy.ndarray): Initial parameter of theta. Its shape must be (len(categories), max(categories)).
    """

    def __init__(self, categories, alpha=1.5, init_delta=1., Delta_max=np.inf, init_theta=None):

        self.p_model = Categorical(categories)

        if init_theta is not None:
            self.p_model.theta = init_theta

        self.N = np.sum(categories - 1)
        self.delta = init_delta
        self.Delta = 1.
        self.Delta_max = np.inf
        self.alpha = alpha
        self.gamma = 0.
        self.s = np.zeros(self.N)

    def get_delta(self):
        return self.delta/self.Delta

    def sampling(self):
        return self.p_model.sampling()

    def update(self, Ms, losses, range_restriction=True):
        delta = self.get_delta()
        beta = delta * self.N**-0.5

        u, idx = self.utility(losses)
        mu_W, var_W = u.mean(), u.var()
        if var_W == 0:
            return

        ngrad = np.mean((u - mu_W)[:, np.newaxis, np.newaxis] * (Ms[idx] - self.p_model.theta), axis=0)

        # Too small natural gradient leads ngnorm to 0.
        if (np.abs(ngrad) < 1e-18).all():
            print('skip update')
            return

        s_latter = []
        for i, K in enumerate(self.p_model.C):
            theta_i = self.p_model.theta[i, :K - 1]
            theta_K = self.p_model.theta[i, K - 1]
            s_i = 1/np.sqrt(theta_i) * ngrad[i, :K - 1]
            s_i += np.sqrt(theta_i) * ngrad[i, :K - 1].sum() / (theta_K + np.sqrt(theta_K))
            s_latter += list(s_i)
        s_latter = np.array(s_latter)

        ngnorm = np.sqrt(np.sum(s_latter**2))
        dp = ngrad/ngnorm
        assert not np.isnan(dp).any(), (ngrad, ngnorm)

        self.p_model.theta += delta * dp

        for i in range(self.p_model.d):
            ci = self.p_model.C[i]

            # Constraint for theta (minimum value of theta and sum of theta = 1.0)
            theta_min = 1. / (self.p_model.valid_d * (ci - 1)) if range_restriction and ci > 1 else 0.
            self.p_model.theta[i, :ci] = np.maximum(self.p_model.theta[i, :ci], theta_min)
            theta_sum = self.p_model.theta[i, :ci].sum()
            tmp = theta_sum - theta_min * ci
            self.p_model.theta[i, :ci] -= (theta_sum - 1.) * (self.p_model.theta[i, :ci] - theta_min) / tmp

            # Ensure the summation to 1
            self.p_model.theta[i, :ci] /= self.p_model.theta[i, :ci].sum()

        self.s = (1 - beta)*self.s + np.sqrt(beta*(2 - beta))*s_latter/ngnorm
        self.gamma = (1 - beta)**2 * self.gamma + beta*(2 - beta)
        self.Delta *= np.exp(beta * (self.gamma - np.dot(self.s, self.s)/self.alpha))
        self.Delta = np.minimum(self.Delta, self.Delta_max)


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
