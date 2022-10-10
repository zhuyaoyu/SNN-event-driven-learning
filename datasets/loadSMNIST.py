import os
import torchvision.datasets
import torchvision.transforms as transforms


def trans(data):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data = trans(data)
    C, H, W = data.shape
    data = data.permute(2,0,1).reshape(W,C,H,1)
    return data


def get_smnist(data_path, network_config):
    print("loading S-MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    trainset = torchvision.datasets.MNIST(data_path, train=True, transform=trans, download=True)
    testset = torchvision.datasets.MNIST(data_path, train=False, transform=trans, download=True)

    return trainset, testset
