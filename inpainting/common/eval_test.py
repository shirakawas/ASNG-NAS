import os
import pandas as pd

import torch
from torch import nn

import common.utils as util
import scipy.misc as spmi


def save_img(img_np, file_name, out_dir='./'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    image = img_np.copy().transpose(1, 2, 0)
    # for gray image
    if img_np.shape[2] == 1:
        image = image[:, :, 0]
    # Revise values to save the matrix as png image
    image[image > 1.] = 1.
    image[image < 0.] = 0.
    spmi.toimage(image, cmin=0, cmax=1).save(out_dir + file_name)


def evaluate(model, test_data, corrupt_func, gpu_id=0, batchsize=64, img_out_dir=None):
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, batchsize, shuffle=False)

    if img_out_dir is not None:
        if not os.path.exists(img_out_dir):
            os.makedirs(img_out_dir)

    df = pd.DataFrame([], columns=['index', 'MLE_MSE', 'MLE_PSNR', 'MLE_SSIM'])
    best_img = []
    worst_img = []
    rank_num = 100

    with torch.no_grad():
        vals = {'MLE_MSE': 0., 'MLE_PSNR': 0., 'MLE_SSIM': 0.}
        loss_func = nn.MSELoss()
        i = 0
        for X, _ in test_loader:
            if gpu_id >= 0:
                X = X.cuda(gpu_id)
            in_img = corrupt_func(X)

            # MLE prediction
            out_mle = model.forward_mle(in_img)

            for j, (org, in_x) in enumerate(zip(X, in_img)):
                # Compute the evaluation measures
                mse = loss_func(out_mle[j], org).item()
                psnr = util.compute_PSNR(out_mle[j].cpu().numpy(), org.cpu().numpy())
                ssim = util.compute_SSIM(out_mle[j].cpu().numpy(), org.cpu().numpy())
                vals['MLE_MSE'] += mse / len(test_data)
                vals['MLE_PSNR'] += psnr / len(test_data)
                vals['MLE_SSIM'] += ssim / len(test_data)

                if img_out_dir is not None:
                    df.loc[i] = [i, mse, psnr, ssim]
                    if len(best_img) < rank_num:
                        best_img.append([i, psnr, in_x.cpu().numpy(), out_mle[j].cpu().numpy(), org.cpu().numpy()])
                    elif psnr > best_img[-1][1]:
                        best_img[-1] = [i, psnr, in_x.cpu().numpy(), out_mle[j].cpu().numpy(), org.cpu().numpy()]
                    best_img.sort(key=lambda x: x[1], reverse=True)

                    if len(worst_img) < rank_num:
                        worst_img.append([i, psnr, in_x.cpu().numpy(), out_mle[j].cpu().numpy(), org.cpu().numpy()])
                    elif psnr < worst_img[-1][1]:
                        worst_img[-1] = [i, psnr, in_x.cpu().numpy(), out_mle[j].cpu().numpy(), org.cpu().numpy()]
                    worst_img.sort(key=lambda x: x[1])
                    i += 1

    if img_out_dir is not None:
        df.to_csv(img_out_dir + 'evaluation.csv', sep=',', header=True, index=False)
        # Save images (best and worst 100 images)
        for i in range(rank_num):
            save_img(best_img[i][2], 'best_rank{:03d}_input_{:05d}.png'.format(i+1, best_img[i][0]), out_dir=img_out_dir)
            save_img(best_img[i][3], 'best_rank{:03d}_mle_out_{:05d}.png'.format(i+1, best_img[i][0]), out_dir=img_out_dir)
            save_img(best_img[i][4], 'best_rank{:03d}_gt_{:05d}.png'.format(i + 1, best_img[i][0]), out_dir=img_out_dir)
            save_img(worst_img[i][2], 'worst_rank{:03d}_input_{:05d}.png'.format(i + 1, worst_img[i][0]), out_dir=img_out_dir)
            save_img(worst_img[i][3], 'worst_rank{:03d}_mle_out_{:05d}.png'.format(i + 1, worst_img[i][0]), out_dir=img_out_dir)
            save_img(worst_img[i][4], 'worst_rank{:03d}_gt_{:05d}.png'.format(i + 1, worst_img[i][0]), out_dir=img_out_dir)

    return vals
