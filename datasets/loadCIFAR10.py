import os
import torchvision
import torchvision.transforms as transforms
from datasets.utils import packaging_class
import PIL


def get_cifar10(data_path, network_config):
    print("loading CIFAR10")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomResizedCrop(32, scale=(0.75,1.0), interpolation=PIL.Image.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

    return trainset, testset
