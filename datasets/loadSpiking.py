import os
import torch
import torchvision.transforms as transforms
import PIL
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets import split_to_train_test_set, RandomTemporalDelete
import numpy as np
from datasets.utils import function_nda, packaging_class
import global_v as glv



def get_dataset(dataset_func, data_path, network_config, transform_train=None, transform_test=None):
    T = network_config['n_steps']
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    trainset = dataset_func(data_path, data_type='frame', frames_number=T, split_by='number', train=True)
    testset = dataset_func(data_path, data_type='frame', frames_number=T, split_by='number', train=False)
    trainset, testset = packaging_class(trainset, transform_train), packaging_class(testset, transform_test)
    return trainset, testset


def get_nmnist(data_path, network_config):
    return get_dataset(NMNIST, data_path, network_config)


def trans_t(data):
    # print(data.shape)
    # exit(0)
    data = transforms.RandomResizedCrop(128, scale=(0.7, 1.0), interpolation=PIL.Image.NEAREST)(data)
    resize = transforms.Resize(size=(48, 48))  # 48 48
    data = resize(data).float()
    flip = np.random.random() > 0.5
    if flip:
        data = torch.flip(data, dims=(3,))
    data = function_nda(data)
    return data.float()


def trans(data):
    resize = transforms.Resize(size=(48, 48))  # 48 48
    data = resize(data).float()
    return data.float()


def get_dvs128_gesture(data_path, network_config):
    T = network_config['t_train']
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.5, 1.0), interpolation=PIL.Image.NEAREST),
        transforms.RandomHorizontalFlip(),
        RandomTemporalDelete(T_remain=T, batch_first=False),
    ])
    return get_dataset(DVS128Gesture, data_path, network_config, transform_train)
    # return get_dataset(DVS128Gesture, data_path, network_config, trans_t, trans)


def get_cifar10_dvs(data_path, network_config):
    T = network_config['n_steps']
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    dataset = CIFAR10DVS(data_path, data_type='frame', frames_number=T, split_by='number')
    trainset, testset = split_to_train_test_set(train_ratio=0.9, origin_dataset=dataset, num_classes=10)

    trainset, testset = packaging_class(trainset, trans_t), packaging_class(testset, trans)
    return trainset, testset
