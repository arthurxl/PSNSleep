import numpy as np
import torch
import torch.nn as nn
import random


class MixUp(nn.Module):
    def __init__(self, args,ratio=0.4,mask_size=100):
        super().__init__()
        self.args=args
        self.ratio = ratio
        self.mask_size=mask_size
        self.m_sum=self.mb_init(self.mask_size)

    @torch.no_grad()
    def mb_init(self, mask_size):
        m_mul = torch.ones((1, 1, 3000))
        xx = random.sample(range(0, 3000 - 1), mask_size)
        for i in (1300,1800):
            m_mul[0][0][i] = 0
        m_sum = torch.repeat_interleave(m_mul, repeats=self.args.batch_size, dim=0)
        return m_sum

    def scaling(x, sigma=1.1):
        factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2])).to(x.get_device())
        ai = []
        for i in range(x.shape[1]):
            xi = x[:, i, :]
            ai.append(xi*factor.unsqueeze(1))
        return torch.cat([ai], axis=1)

    def forward(self, x):

        xa = x
        xb = torch.index_select(xa, 0, torch.randperm(x.size(0)).to(x.get_device())) #混合样本生成
        x = self.ratio * xa + (1. - self.ratio) * xb #混合

        return x

def log_mixup_exp(xa, xb, alpha):
    xa=torch.exp(xa)
    xb=torch.exp(xb)
    x = alpha * xa + (1. - alpha) * xb
    return torch.log(x + torch.finfo(x.dtype).eps)


class MixupBYOLA(nn.Module):
    def __init__(self, ratio=0.4, n_memory=20, log_mixup_exp=True):
        super().__init__()
        self.ratio = ratio
        self.n = n_memory
        self.log_mixup_exp = log_mixup_exp
        self.memory_bank = []
    @torch.no_grad()
    def mb_init(self, z):
        for i in range(self.n):
            self.memory_bank.append(z)

    @torch.no_grad()
    def mb_update(self,z):
        index=random.randint(0,self.n-1)
        self.memory_bank[index]=z

    def forward(self, x):
        # mix random
        alpha = self.ratio * np.random.random()
        if self.memory_bank:


            # get z as a mixing background sound
            z = self.memory_bank[np.random.randint(len(self.memory_bank))]
            # mix them
            mixed = log_mixup_exp(x, z, 1. - alpha) if self.log_mixup_exp \
                    else alpha * z + (1. - alpha) * x

        else:
            self.mb_init(x)
            mixed = x
        #update memory bank
        self.memory_bank = (self.memory_bank + [x])[-self.n:]
        self.mb_update(x)




        return mixed.to(torch.float)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(ratio={self.ratio},n={self.n}'
        format_string += f',log_mixup_exp={self.log_mixup_exp})'
        return format_string


class MixGaussianNoise():
    """Gaussian Noise Mixer.
    This interpolates with random sample, unlike Mixup.
    """

    def __init__(self, ratio=0.3):
        self.ratio = ratio

    def forward(self, lms):
        x = lms.exp()

        lambd = self.ratio * np.random.rand()
        z = torch.normal(0, lambd, x.shape).exp()
        mixed = (1 - lambd) * x + z + torch.finfo(x.dtype).eps

        return mixed.log()

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(ratio={self.ratio})'
        return format_string

class RunningMean:
    """Running mean calculator for arbitrary axis configuration."""

    def __init__(self, axis):
        self.n = 0
        self.axis = axis

    def put(self, x):
        # https://math.stackexchange.com/questions/106700/incremental-averageing
        if self.n == 0:
            self.mu = x.mean(self.axis, keepdims=True)
        else:
            self.mu += (x.mean(self.axis, keepdims=True) - self.mu) / self.n
        self.n += 1

    def __call__(self):
        return self.mu

    def __len__(self):
        return self.n


class RunningVariance:
    """Calculate mean/variance of tensors online.
    Thanks to https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """

    def __init__(self, axis, mean):
        self.update_mean(mean)
        self.s2 = RunningMean(axis)

    def update_mean(self, mean):
        self.mean = mean

    def put(self, x):
        self.s2.put((x - self.mean) **2)

    def __call__(self):
        return self.s2()

    def std(self):
        return np.sqrt(self())


class RunningNorm(nn.Module):
    """Online Normalization using Running Mean/Std.

    This module will only update the statistics up to the specified number of epochs.
    After the `max_update_epochs`, this will normalize with the last updated statistics.

    Args:
        epoch_samples: Number of samples in one epoch
        max_update_epochs: Number of epochs to allow update of running mean/variance.
        axis: Axis setting used to calculate mean/variance.
    """

    def __init__(self, epoch_samples, max_update_epochs=10, axis=[1, 2]):
        super().__init__()
        self.max_update = epoch_samples * max_update_epochs
        self.ema_mean = RunningMean(axis)
        self.ema_var = RunningVariance(axis, 0)

    def forward(self, image):
        if len(self.ema_mean) < self.max_update:
            self.ema_mean.put(image)
            self.ema_var.update_mean(self.ema_mean())
            self.ema_var.put(image)
            self.mean = self.ema_mean()
            self.std = torch.clamp(self.ema_var.std(), torch.finfo().eps, torch.finfo().max)
        return ((image - self.mean) / self.std)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(max_update={self.max_update},axis={self.ema_mean.axis})'
        return format_string


if __name__ == '__main__':
    data=np.random.random([2,2,5])
    print(data)
    data_aug=gaussian_noise(data)
    print(data_aug)