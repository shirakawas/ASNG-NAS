import numpy as np
import torch
import torchvision.transforms as transforms

from skimage.measure import compare_ssim


def data_transforms(img_size=64):
    train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize(size=img_size),
             transforms.ToTensor(),
             ]
            )

    valid_transform = transforms.Compose(
            [transforms.Resize(size=img_size),
             transforms.ToTensor(),
             ]
            )
    return train_transform, valid_transform


def compute_PSNR(out_img, ground_truth, max_value=1):
    img = out_img.copy()
    img[img > 1.] = 1.
    img[img < 0.] = 0.
    gt = ground_truth.copy()
    gt[gt > 1.] = 1.
    gt[gt < 0.] = 0.
    mse = np.mean((img - gt)**2)
    return 20 * np.log10(max_value) - 10 * np.log10(mse)


def compute_SSIM(out_img, ground_truth):
    img = out_img.copy().transpose(1, 2, 0)
    img[img > 1.] = 1.
    img[img < 0.] = 0.
    gt = ground_truth.copy().transpose(1, 2, 0)
    gt[gt > 1.] = 1.
    gt[gt < 0.] = 0.
    return compare_ssim(img, gt, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, multichannel=True,
                        data_range=1.)


class RandomPixelMasking(object):
    def __init__(self, fraction_masked=0.8):
        self.fraction_masked = fraction_masked

    def __call__(self, img):
        batch, _, img_height, img_width = img.shape
        mask = torch.ones((batch, 1,  img_height, img_width), device=img.get_device())
        rnd = torch.rand(batch, 1, img_height, img_width, device=img.get_device())
        mask[rnd < self.fraction_masked] = 0.
        return img * mask


class RandomHalfMasking(object):
    def __init__(self, *args):
        pass

    def __call__(self, img):
        batch, _, img_height, img_width = img.shape
        mask = torch.ones_like(img)
        for i in range(batch):
            rnd = np.random.randint(low=0, high=4)
            if rnd == 0:
                # left
                mask[i, :, :, 0:int(img_width / 2)] = 0.
            elif rnd == 1:
                # up
                mask[i, :, 0:int(img_height / 2), :] = 0.
            elif rnd == 2:
                # right
                mask[i, :, :, int(img_width / 2):img_width] = 0.
            else:
                # bottom
                mask[i, :, int(img_height / 2):img_height, :] = 0.
        return img * mask


class CenterMasking(object):
    def __init__(self, scale=0.25):
        self.scale = scale

    def __call__(self, img):
        scale = self.scale
        _, _, img_height, img_width = img.shape
        mask_top, mask_bottom = int(img_height * scale), int(img_height * (1. - scale))
        mask_left, mask_right = int(img_width * scale), int(img_width * (1. - scale))
        mask = torch.ones_like(img)
        mask[:, :, mask_top:mask_bottom, mask_left:mask_right] = 0.
        return img * mask


class AdditionalGaussianNoising(object):
    def __init__(self, sigmas):
        self.sigmas = sigmas

    def __call__(self, img):
        sigma = self.sigmas[torch.randint(len(self.sigmas))]
        noise = torch.randn_like(img) * (sigma / 255.)
        noisy_img = img + noise
        noisy_img[noisy_img > 1.] = 1.
        noisy_img[noisy_img < 0.] = 0.
        return noisy_img
