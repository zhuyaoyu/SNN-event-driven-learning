import os
import torch
from torch.utils.data import Dataset
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS


class spiking_dataset(Dataset):
    def __init__(self, data_path, dataset_func, T, train, transform=None):
        self.transform = transform
        self.dataset = dataset_func(data_path, data_type='frame', frames_number=T, split_by='number', train=train)

    def __getitem__(self, index):
        data, label = self.dataset[index]
        if self.transform:
            data = self.transform(data).type(torch.float32)
        else:
            data = torch.FloatTensor(data)

        return data, label

    def __len__(self):
        return len(self.dataset)


def get_dataset(dataset_func, data_path, network_config):
    T = network_config['n_steps']
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    trainset = spiking_dataset(data_path, dataset_func, T=T, train=True)
    testset = spiking_dataset(data_path, dataset_func, T=T, train=False)
    return trainset, testset


def get_nmnist(data_path, network_config):
    return get_dataset(NMNIST, data_path, network_config)


def get_dvs128_gesture(data_path, network_config):
    return get_dataset(DVS128Gesture, data_path, network_config)


def get_cifar10_dvs(data_path, network_config):
    return get_dataset(CIFAR10DVS, data_path, network_config)
