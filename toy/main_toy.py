
import numpy as np
import torch

from asng import AdaptiveSNG
from sng import SNG
from adam import Adam
from fxc import fxc1
from opt import run


def experiment(alg='ASNG', eta_x=0.1, eta_theta_factor=0., alpha=1.5, K=5, D=30, maxite=100000, log_file='log.csv'):
    nc = (K-1) * D
    f = fxc1(K, D, noise=True)
    categories = K * np.ones(D, dtype=np.int)

    if alg == 'ASNG':
        opt_theta = AdaptiveSNG(categories, alpha=alpha, delta_init=nc**-eta_theta_factor)
    elif alg == 'SNG':
        opt_theta = SNG(categories, delta_init=nc**-eta_theta_factor)
    elif alg == 'Adam':
        opt_theta = Adam(categories, alpha=nc**-eta_theta_factor, beta1=0.9, beta2=0.999)
    else:
        print('invalid algorithm!')
        return

    optimizer_x = torch.optim.SGD(f.parameters(), lr=eta_x, momentum=0.9, weight_decay=0., nesterov=False)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_x, maxite)

    print('{}, eta_x={}, eta_theta_factor={} alpha={}'.format(alg, eta_x, eta_theta_factor, alpha))
    run(f, opt_theta, optimizer_x, lr_scheduler=lr_scheduler, maxite=maxite, dispspan=100, log_file=log_file)


if __name__ == '__main__':
    # alg: 'ASNG' or 'SNG' or 'Adam'
    # eta_x: step-size for x
    # eta_theta_factor: step-size for $\theta$ is 4n_c^(-eta_theta_factor)$
    experiment(alg='ASNG', eta_x=0.05, eta_theta_factor=0., alpha=1.5, K=5, D=30, maxite=100000, log_file='log.csv')
