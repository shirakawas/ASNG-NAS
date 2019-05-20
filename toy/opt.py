
import numpy as np
import csv
import torch


def run(f, optimizer_theta, optimizer_x, lr_scheduler=None, maxite=1e5, dispspan=100, logspan=100, log_file=None):
    t = 1
    gloss = np.inf

    if log_file is not None:
        with open(log_file, 'w') as fp:
            writer = csv.writer(fp, lineterminator='\n')
            writer.writerow(['iteratoin', 'eval', 'gloss'] + optimizer_theta.log_header(theta_log=False))

    while t <= maxite:
        f.train()
        if lr_scheduler is not None:
            lr_scheduler.step()

        lam = optimizer_theta.get_lam()
        losses = np.array([])
        general_losses = np.array([])

        # sampling and evaluation
        c = np.array([optimizer_theta.sampling() for _ in range(lam)], dtype=np.int32)
        optimizer_x.zero_grad()
        loss, gloss = f(torch.Tensor(c))
        losses = np.append(losses, loss.detach().numpy())
        general_losses = np.append(general_losses, gloss.detach().numpy())
        mean_loss = loss.mean()

        # update x
        mean_loss.backward()
        optimizer_x.step()

        # sampling and evaluation
        c = np.array([optimizer_theta.sampling() for _ in range(lam)], dtype=np.int32)
        loss, gloss = f(torch.Tensor(c))
        losses = np.append(losses, loss.detach().numpy())
        general_losses = np.append(general_losses, gloss.detach().numpy())

        # update c
        optimizer_theta.update(c, losses[-lam:])

        # logging
        gloss = np.min(general_losses)
        if t % dispspan == 0 or t == 1 or t + 1 == maxite:
            print('ite: {} gloss {:f} theta_convergence: {:.4f} theta_mle: {:d} delta: {}'.format(
                t, gloss, optimizer_theta.theta.max(axis=1).mean(), int(optimizer_theta.mle()[:, 0].sum()),
                optimizer_theta.get_delta()))
        if log_file is not None and (t % logspan == 0 or t == 1):
            with open(log_file, 'a') as fp:
                writer = csv.writer(fp, lineterminator='\n')
                writer.writerow([t, f.eval, gloss] + optimizer_theta.log(theta_log=False))
        if gloss <= f.target:
            break
        t += 1

    print('ite: {} gloss {:f} theta_convergence: {:.4f} theta_mle: {:d} delta: {}'.format(
        t, gloss, optimizer_theta.theta.max(axis=1).mean(), int(optimizer_theta.mle()[:, 0].sum()),
        optimizer_theta.get_delta()))

    if log_file is not None:
        with open(log_file, 'a') as fp:
            writer = csv.writer(fp, lineterminator='\n')
            writer.writerow([t, f.eval, gloss] + optimizer_theta.log(theta_log=False))

    return gloss <= f.target, t
