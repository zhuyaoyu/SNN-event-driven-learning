import os
import torchvision.datasets
import torchvision.transforms as transforms
import torch


def get_fashionmnist(data_path, network_config):
    print("loading Fashion MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    transform_train = transforms.Compose([
        # transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.FashionMNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.FashionMNIST(data_path, train=False, transform=transform_test, download=True)
    
    return trainset, testset

