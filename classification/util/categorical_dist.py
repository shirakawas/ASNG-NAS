#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class Categorical(object):
    """
    Categorical distribution for categorical variables parametrized by :math:`\\{ \\theta \\}_{i=1}^{(d \\times K)}`.

    :param categories: the numbers of categories
    :type categories: array_like, shape(d), dtype=int
    """
    def __init__(self, categories):
        self.d = len(categories)
        self.C = categories
        self.Cmax = np.max(categories)
        self.theta = np.zeros((self.d, self.Cmax))
        # initialize theta by 1/C for each dimensions
        for i in range(self.d):
            self.theta[i, :self.C[i]] = 1./self.C[i]
        # pad zeros to unused elements
        for i in range(self.d):
            self.theta[i, self.C[i]:] = 0.
        # number of valid parameters
        self.valid_param_num = int(np.sum(self.C - 1))
        # valid dimension size
        self.valid_d = len(self.C[self.C > 1])

    def sampling_lam(self, lam):
        """
        Draw :math:`\\lambda` samples from the categorical distribution.
        :param int lam: sample size :math:`\\lambda`
        :return: sampled variables from the categorical distribution (one-hot representation)
        :rtype: array_like, shape=(lam, d, Cmax), dtype=bool
        """
        rand = np.random.rand(lam, self.d, 1)    # range of random number is [0, 1)
        cum_theta = self.theta.cumsum(axis=1)    # (d, Cmax)
        X = (cum_theta - self.theta <= rand) & (rand < cum_theta)
        return X

    def sampling(self):
        """
        Draw a sample from the categorical distribution.
        :return: sampled variables from the categorical distribution (one-hot representation)
        :rtype: array_like, shape=(d, Cmax), dtype=bool
        """
        rand = np.random.rand(self.d, 1)    # range of random number is [0, 1)
        cum_theta = self.theta.cumsum(axis=1)    # (d, Cmax)

        # x[i, j] becomes 1 iff cum_theta[i, j] - theta[i, j] <= rand[i] < cum_theta[i, j]
        x = (cum_theta - self.theta <= rand) & (rand < cum_theta)
        return x

    def mle(self):
        """
        Return the most likely categories.
        :return: categorical variables (one-hot representation)
        :rtype: array_like, shape=(d, Cmax), dtype=bool
        """
        m = self.theta.argmax(axis=1)
        x = np.zeros((self.d, self.Cmax))
        for i, c in enumerate(m):
            x[i, c] = 1
        return x

    def loglikelihood(self, X):
        """
        Calculate log likelihood.

        :param X: samples (one-hot representation)
        :type X: array_like, shape=(lam, d, maxK), dtype=bool
        :return: log likelihoods
        :rtype: array_like, shape=(lam), dtype=float
        """
        return (X * np.log(self.theta)).sum(axis=2).sum(axis=1)

    def log_header(self):
        header_list = []
        for i in range(self.d):
            header_list += ['theta%d_%d' % (i, j) for j in range(self.C[i])]
        return header_list

    def log(self):
        theta_list = []
        for i in range(self.d):
            theta_list += ['%f' % self.theta[i, j] for j in range(self.C[i])]
        return theta_list

    def load_theta_from_log(self, theta):
        self.theta = np.zeros((self.d, self.Cmax))
        k = 0
        for i in range(self.d):
            for j in range(self.C[i]):
                self.theta[i, j] = theta[k]
                k += 1
