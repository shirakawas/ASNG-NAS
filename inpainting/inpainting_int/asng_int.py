#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class AdaptiveSNG:
    """
    Adaptive Stochastic Natural Gradient for Categorical Distribution
    """
    def __init__(self, categories, integers, alpha=1.5, delta_init=1., lam=2, Delta_max=np.inf, init_theta=None):

        self.n_cat = np.sum(categories - 1)
        self.n_int = 2 * len(integers)
        self.n_theta = self.n_cat + self.n_int

        # Categorical distribution
        self.d_cat = len(categories)
        self.C = categories
        self.Cmax = np.max(categories)
        self.theta_cat = np.zeros((self.d_cat, self.Cmax))
        # initialize theta by 1/C for each dimensions
        for i in range(self.d_cat):
            self.theta_cat[i, :self.C[i]] = 1. / self.C[i]
        # pad zeros to unused elements
        for i in range(self.d_cat):
            self.theta_cat[i, self.C[i]:] = 0.
        # valid dimension size
        self.valid_d_cat = len(self.C[self.C > 1])

        # Normal distribution
        self.d_int = len(integers)
        self.int_min = np.array(integers)[:, 0]
        self.int_max = np.array(integers)[:, 1]
        self.int_std_max = (self.int_max-self.int_min)/2.
        self.int_std_min = 1./4.
        # initialize theta
        self.theta_int = np.zeros((2, self.d_int))
        self.theta_int[0] = (self.int_max + self.int_min) / 2.
        self.theta_int[1] = ((self.int_max + self.int_min) / 2.)**2 + self.int_std_max**2

        if init_theta is not None:
            self.theta_cat, self.theta_int = init_theta[0], init_theta[1]

        # Adaptive SG
        self.alpha = alpha  # threshold for adaptation
        self.delta_init = delta_init
        self.lam = lam  # lambda_theta
        self.Delta_max = Delta_max  # maximum Delta (can be np.inf)

        self.Delta = 1.
        self.gamma = 0.0  # correction factor
        self.s = np.zeros(self.n_theta)  # averaged stochastic natural gradient
        self.delta = self.delta_init / self.Delta
        self.eps = self.delta

    def get_lam(self):
        return self.lam

    def get_delta(self):
        return self.delta

    def sampling(self, lam):
        # Draw a sample from the categorical distribution (one-hot)
        rand = np.random.rand(lam, self.d_cat, 1)  # range of random number is [0, 1)
        cum_theta = self.theta_cat.cumsum(axis=1)  # (d, Cmax)
        # x[i, j] becomes 1 if cum_theta[i, j] - theta[i, j] <= rand[i] < cum_theta[i, j]
        c_cat = (cum_theta - self.theta_cat <= rand) & (rand < cum_theta)

        c_int = np.empty((lam+1, self.d_int))
        avg = self.theta_int[0]
        std = np.sqrt(self.theta_int[1] - avg ** 2)
        for i in range(int(np.round(lam/2))):
            # Draw a sample from the normal distribution
            # symmetric sampling
            Z = np.random.randn(self.d_int)
            c_int[2*i] = avg + std * Z
            c_int[2*i+1] = avg - std * Z

        return c_cat, c_int[:lam]

    def mle(self):
        """
        Get most likely categorical variables (one-hot)
        """
        c_cat = self.theta_cat.argmax(axis=1)
        T = np.zeros((self.d_cat, self.Cmax))
        for i, c in enumerate(c_cat):
            T[i, c] = 1
        return T, self.theta_int[0]

    def update(self, c, fxc, range_restriction=True):
        self.delta = self.delta_init / self.Delta
        beta = self.delta / (self.n_theta ** 0.5)

        aru, idx = self.utility(fxc)
        if np.all(aru == 0):
            # If all the points have the same f-value,
            # nothing happens for theta and breaks.
            # In this case, we skip the rest of the code.
            return

        c_cat, c_int = c[0], c[1]

        # NG for categorical distribution
        ng_cat = np.mean(aru[:, np.newaxis, np.newaxis] * (c_cat[idx] - self.theta_cat), axis=0)

        # sqrt(F) * NG for categorical distribution
        sl = []
        for i, K in enumerate(self.C):
            theta_i = self.theta_cat[i, :K - 1]
            theta_K = self.theta_cat[i, K - 1]
            s_i = 1. / np.sqrt(theta_i) * ng_cat[i, :K - 1]
            s_i += np.sqrt(theta_i) * ng_cat[i, :K - 1].sum() / (theta_K + np.sqrt(theta_K))
            sl += list(s_i)
        sl = np.array(sl)

        # NG for normal distribution
        ng_int1 = np.mean(aru[:, np.newaxis] * (c_int[idx] - self.theta_int[0]), axis=0)
        ng_int2 = np.mean(aru[:, np.newaxis] * (c_int[idx]**2 - self.theta_int[1]), axis=0)
        dpara = np.vstack((ng_int1, ng_int2))

        # sqrt(F) * NG for normal distribution
        avg = self.theta_int[0]
        std = np.sqrt(self.theta_int[1] - avg ** 2)
        eigval = np.zeros((2, self.d_int))
        eigvec1 = np.zeros((2, self.d_int))
        eigvec2 = np.zeros((2, self.d_int))
        # Inverse Fisher (Ordinal)
        fb = 2 * avg
        fc = 4 * avg ** 2 + 2 * std ** 2
        eigval[0, :] = ((1. + fc) + np.sqrt((1 - fc) ** 2 + 4 * fb ** 2)) / 2.
        eigval[1, :] = ((1. + fc) - np.sqrt((1 - fc) ** 2 + 4 * fb ** 2)) / 2.
        mask = np.abs(fb) < 1e-8
        neg_mask = np.logical_not(mask)
        eigvec1[1, mask] = eigvec2[0, mask] = 0.
        eigvec1[0, mask] = eigvec2[1, mask] = 1.
        eigvec1[0, neg_mask] = eigvec2[0, neg_mask] = 1.
        eigvec1[1, neg_mask] = (eigval[0, neg_mask] - 1.) / fb[neg_mask]
        eigvec2[1, neg_mask] = (eigval[1, neg_mask] - 1.) / fb[neg_mask]
        eigvec1 /= np.linalg.norm(eigvec1, axis=0)
        eigvec2 /= np.linalg.norm(eigvec2, axis=0)
        eigval[0, :] *= std ** 2
        eigval[1, :] *= std ** 2

        # sqrt(F) * NG
        fdpara = np.zeros((2, self.d_int))  # sqrt(F) * dtheta
        fdpara[0, :] = eigvec1[0, :] * dpara[0] + eigvec1[1, :] * dpara[1]
        fdpara[1, :] = eigvec2[0, :] * dpara[0] + eigvec2[1, :] * dpara[1]
        fdpara[0, :] /= np.sqrt(eigval[0, :])
        fdpara[1, :] /= np.sqrt(eigval[1, :])
        fdpara = eigvec1 * fdpara[0] + eigvec2 * fdpara[1]

        fnorm_cat = np.sum(sl ** 2)
        fnorm_ord = np.sum(fdpara ** 2)
        fnorm = np.sqrt(fnorm_cat + fnorm_ord)
        self.eps = self.delta / (fnorm + 1e-9)

        # update
        self.theta_cat += self.eps * ng_cat
        self.theta_int += self.eps * dpara

        self.s = (1 - beta) * self.s + np.sqrt(beta * (2 - beta)) * np.hstack((sl, np.ravel(fdpara))) / fnorm
        self.gamma = (1 - beta)**2 * self.gamma + beta*(2 - beta)
        self.Delta *= np.exp(beta * (self.gamma - np.sum(self.s**2) / self.alpha))
        self.Delta = min(self.Delta, self.Delta_max)
        
        # range restriction
        for i in range(self.d_cat):
            ci = self.C[i]
            # Constraint for theta (minimum value of theta and sum of theta = 1.0)
            theta_min = 1. / (self.valid_d_cat * (ci - 1)) if range_restriction and ci > 1 else 0.
            self.theta_cat[i, :ci] = np.maximum(self.theta_cat[i, :ci], theta_min)
            theta_sum = self.theta_cat[i, :ci].sum()
            tmp = theta_sum - theta_min * ci
            self.theta_cat[i, :ci] -= (theta_sum - 1.) * (self.theta_cat[i, :ci] - theta_min) / tmp
            # Ensure the summation to 1
            self.theta_cat[i, :ci] /= self.theta_cat[i, :ci].sum()

        self.theta_int[0] = np.clip(self.theta_int[0], self.int_min, self.int_max)
        self.theta_int[1] = np.clip(self.theta_int[1], self.theta_int[0]**2 + self.int_std_min**2,
                                    self.theta_int[0]**2 + self.int_std_max**2)

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
        header_list = ['delta', 'eps', 'snorm_alpha', 'theta_cat_converge']
        if theta_log:
            for i in range(self.d_cat):
                header_list += ['theta_cat%d_%d' % (i, j) for j in range(self.C[i])]
            header_list += ['theta_int1_%d' % i for i in range(self.d_int)]
            header_list += ['theta_int2_%d' % i for i in range(self.d_int)]
        return header_list

    def log(self, theta_log=False):
        log_list = [self.delta, self.eps, np.dot(self.s, self.s) / self.alpha, self.theta_cat.max(axis=1).mean()]
        if theta_log:
            for i in range(self.d_cat):
                log_list += [self.theta_cat[i, j] for j in range(self.C[i])]
            log_list += [self.theta_int[0, i] for i in range(self.d_int)]
            log_list += [self.theta_int[1, i] for i in range(self.d_int)]
        return log_list

    def load_theta_from_log(self, theta_log):
        self.theta_cat = np.zeros((self.d_cat, self.Cmax))
        k = 0
        for i in range(self.d_cat):
            for j in range(self.C[i]):
                self.theta_cat[i, j] = theta_log[k]
                k += 1

        self.theta_int = np.zeros((2, self.d_int))
        for i in range(self.d_int):
            self.theta_cat[0, i] = theta_log[k]
            k += 1
        for i in range(self.d_int):
            self.theta_cat[1, i] = theta_log[k]
            k += 1
