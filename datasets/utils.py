import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import PIL
import math
import numpy as np


class packaging_class(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        data = torch.FloatTensor(data)
        if self.transform:
            data = self.transform(data)

        return data, label

    def __len__(self):
        return len(self.dataset)


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h = img.size(2)
        w = img.size(3)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


def function_nda(data, M=1, N=2):
    c = 15 * N
    rotate_tf = transforms.RandomRotation(degrees=c)
    e = 8 * N
    cutout_tf = Cutout(length=e)

    def roll(data, N=1):
        a = N * 2 + 1
        off1 = np.random.randint(-a, a + 1)
        off2 = np.random.randint(-a, a + 1)
        return torch.roll(data, shifts=(off1, off2), dims=(2, 3))

    def rotate(data, N):
        return rotate_tf(data)

    def cutout(data, N):
        return cutout_tf(data)

    transforms_list = [roll, rotate, cutout]
    sampled_ops = np.random.choice(transforms_list, M)
    for op in sampled_ops:
        data = op(data, N)
    return data


def TTFS(data, T):
    # data: C*H*W
    # output: T*C*H*W
    C, H, W = data.shape
    low, high = torch.min(data), torch.max(data)
    data = ((data - low) / (high - low) * T).long()
    # T --> T-1
    data = torch.clip(data, 0, T - 1)
    res = torch.zeros(T, C, H, W, device=data.device)
    return res.scatter_(0, data.unsqueeze(0), 1)
