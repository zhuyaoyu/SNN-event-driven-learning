import os
import sys
import shutil

import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from network_parser import parse
from datasets import loadMNIST, loadCIFAR10, loadCIFAR100, loadFashionMNIST, loadNMNIST_Spiking
import cnns
from utils import learningStats
import layers.losses as losses
import numpy as np
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
import global_v as glv

from sklearn.metrics import confusion_matrix
import argparse


log_interval = 100
multigpu = False


def get_loss(network_config, err, outputs, labels):
    if network_config['loss'] == "count":
        # set target signal
        desired_count = network_config['desired_count']
        undesired_count = network_config['undesired_count']
        targets = torch.ones_like(outputs[0]) * undesired_count
        for i in range(len(labels)):
            targets[i, labels[i]] = desired_count
        loss = err.spike_count(outputs, targets)
    elif network_config['loss'] == "kernel":
        targets = torch.zeros_like(outputs)
        device = torch.device(glv.rank)
        if T >= 8:
            desired_spikes = torch.tensor([0, 1], device=device).repeat(T // 2)
            if T % 2 == 1:
                desired_spikes = torch.cat([torch.zeros(1, device=device), desired_spikes])
        else:
            desired_spikes = torch.ones(T, device=device)
            desired_spikes[0] = 0
        desired_spikes = desired_spikes.view(T, 1, 1)
        for i in range(len(labels)):
            targets[..., i, labels[i]] = desired_spikes
        loss = err.spike_kernel(outputs, targets)
    elif network_config['loss'] == "TET":
        # set target signal
        loss = err.spike_TET(outputs, labels)
    else:
        raise Exception('Unrecognized loss function.')

    return loss.to(glv.rank)


def train(network, trainloader, opti, epoch, states, err):
    train_loss, correct, total = 0, 0, 0
    cnt_oneof, cnt_unique = 0, 0
    network_config, layers_config = glv.network_config, glv.layers_config
    T = network_config['n_steps']
    n_class = network_config['n_class']
    batch_size = network_config['batch_size']
    scaler = GradScaler()
    start_time = datetime.now()

    forward_time, backward_time, data_time, other_time, glv.time_use = 0, 0, 0, 0, 0
    t0 = datetime.now()
    batch_idx = 0
    for inputs, labels in trainloader:
        torch.cuda.synchronize()
        data_time += (datetime.now() - t0).total_seconds()
        t0 = datetime.now()
        batch_idx += 1

        if len(inputs.shape) < 5:
            inputs = inputs.unsqueeze_(0).repeat(T, 1, 1, 1, 1)
        # forward pass
        labels, inputs = (x.to(glv.rank) for x in (labels, inputs))
        if network_config['amp']:
            with autocast():
                outputs = network(inputs, labels, epoch, True)
                loss = get_loss(network_config, err, outputs, labels)
        else:
            outputs = network(inputs, labels, epoch, True)
            loss = get_loss(network_config, err, outputs, labels)
        assert (len(outputs.shape) == 3)

        torch.cuda.synchronize()
        forward_time += (datetime.now() - t0).total_seconds()
        t0 = datetime.now()
        # backward pass
        opti.zero_grad()
        if network_config['amp']:
            scaler.scale(loss).backward()
            scaler.unscale_(opti)
            clip_grad_norm_(network.parameters(), 1)
            scaler.step(opti)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(network.parameters(), 1)
            opti.step()
        (network.module if multigpu else network).weight_clipper()
        torch.cuda.synchronize()
        backward_time += (datetime.now() - t0).total_seconds()
        t0 = datetime.now()

        glv.outputs_raw *= 1.1 - torch.arange(T, device=torch.device(glv.rank)).reshape(T, 1, 1) / T / 10
        spike_counts = torch.sum(glv.outputs_raw, dim=0).detach()
        predicted = torch.argmax(spike_counts, axis=1)
        train_loss += torch.sum(loss).item()
        total += len(labels)
        correct += (predicted == labels).sum().item()

        states.training.correctSamples = correct
        states.training.numSamples = total
        states.training.lossSum += loss.to('cpu').data.item()

        labels = labels.reshape(-1)
        idx = torch.arange(batch_size, device=torch.device(glv.rank))
        nspike_label = spike_counts[idx, labels]
        cnt_oneof += torch.sum(nspike_label == torch.max(spike_counts, axis=1).values).item()
        spike_counts[idx, labels] -= 1
        cnt_unique += torch.sum(nspike_label > torch.max(spike_counts, axis=1).values).item()
        spike_counts[idx, labels] += 1

        if (not multigpu or dist.get_rank() == 0) and batch_idx % log_interval == 0:
            # if batch_idx % log_interval == 0:
            states.print(epoch, batch_idx, (datetime.now() - start_time).total_seconds())
            print('Time consumed on loading data = %.2f, forward = %.2f, backward = %.2f, other = %.2f'
                  % (data_time, forward_time, backward_time, other_time))
            print('Time consumed on manual conv grad = %.2f' % (glv.time_use))
            data_time, forward_time, backward_time, other_time, glv.time_use = 0, 0, 0, 0, 0

            grad_std = dict()
            for idx, layer in enumerate((network.module if multigpu else network).layers):
                if hasattr(layer, 'weight') and layer.weight.requires_grad:
                    gradd = layer.weight.grad.reshape(-1)
                    grad_std[layer.name] = gradd.std()
            print('grad std: ', grad_std)

            spike_cnt = dict()
            for i in range(6):
                spike_cnt['nspike=' + str(i)] = torch.sum(spike_counts > i - 1) - torch.sum(
                    spike_counts > i) if i < 5 else torch.sum(spike_counts > 4)
            print('Spike_cnt is', spike_cnt)

            avg_oneof, avg_unique = cnt_oneof / (batch_size * batch_idx), cnt_unique / (batch_size * batch_idx)
            print('Percentage of partially right = %.2f%%, entirely right = %.2f%%' % (avg_oneof * 100, avg_unique * 100))
            print()
        torch.cuda.synchronize()
        other_time += (datetime.now() - t0).total_seconds()
        t0 = datetime.now()

    acc = correct / total
    train_loss = train_loss / total

    return acc, train_loss


def test(network, testloader, epoch, states, log_dir):
    global best_acc
    correct = 0
    total = 0
    network_config = glv.network_config
    T = network_config['n_steps']
    n_class = network_config['n_class']
    time = datetime.now()
    y_pred = []
    y_true = []
    batch_idx = 0
    for inputs, labels in testloader:
        batch_idx += 1
        if len(inputs.shape) < 5:
            inputs = inputs.unsqueeze_(0).repeat(T, 1, 1, 1, 1)
        # forward pass
        labels = labels.to(glv.rank)
        inputs = inputs.to(glv.rank)
        with torch.no_grad():
            outputs = network(inputs, None, epoch, False)

        outputs *= 1.1 - torch.arange(T, device=torch.device(glv.rank)).reshape(T, 1, 1) / T / 10
        spike_counts = torch.sum(outputs, dim=0).detach().cpu().numpy()
        predicted = np.argmax(spike_counts, axis=1)
        labels = labels.cpu().numpy()
        y_pred.append(predicted)
        y_true.append(labels)
        total += len(labels)
        correct += (predicted == labels).sum().item()

        states.testing.correctSamples += (predicted == labels).sum().item()
        states.testing.numSamples = total
        if batch_idx % log_interval == 0:
            states.print(epoch, batch_idx, (datetime.now() - time).total_seconds())
    print()

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    nums = np.bincount(y_true)
    confusion = confusion_matrix(y_true, y_pred, labels=np.arange(n_class)) / nums.reshape(-1, 1)

    test_acc = correct / total

    state = {
        'net': (network.module if multigpu else network).state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(log_dir, 'last.pth'))

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(state, os.path.join(log_dir, 'best.pth'))
    return test_acc, confusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', action='store', dest='config', help='The path of config file')
    parser.add_argument('-checkpoint', action='store', dest='checkpoint',
                        help='The path of checkpoint, if use checkpoint')
    parser.add_argument('-seed', type=int, default=3, help='random seed (default: 3)')
    parser.add_argument('-dist', type=str, default="nccl", help='distributed data parallel backend')
    parser.add_argument('--local_rank', type=int, default=-1)
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if args.config is None:
        raise Exception('Unrecognized config file.')
    else:
        config_path = args.config

    params = parse(config_path)

    # check GPU
    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)
    # set GPU
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend=args.dist)
        glv.rank = args.local_rank
        multigpu = True
    else:
        glv.rank = 0
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    glv.init(params['Network'], params['Layers'])

    data_path = os.path.expanduser(params['Network']['data_path'])
    dataset_func = {"MNIST": loadMNIST.get_mnist,
                    "NMNIST_Spiking": loadNMNIST_Spiking.get_nmnist,
                    "FashionMNIST": loadFashionMNIST.get_fashionmnist,
                    "CIFAR10": loadCIFAR10.get_cifar10,
                    "CIFAR100": loadCIFAR100.get_cifar100}
    try:
        trainset, testset = dataset_func[params['Network']['dataset']](data_path, params['Network'])
    except:
        raise Exception('Unrecognized dataset name.')
    batch_size = params['Network']['batch_size']
    if multigpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4,
                                                   sampler=train_sampler, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    net = cnns.Network(list(train_loader.dataset[0][0].shape)).to(glv.rank)
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
        net.load_state_dict(checkpoint['net'])
        epoch_start = checkpoint['epoch'] + 1
        print('Network loaded.')
        print(f'Start training from epoch {epoch_start}.')
    else:
        inputs = torch.stack([train_loader.dataset[i][0] for i in range(batch_size)], dim=0).to(glv.rank)
        T = params['Network']['n_steps']
        if len(inputs.shape) < 5:
            inputs = inputs.unsqueeze_(0).repeat(T, 1, 1, 1, 1)
        cnns.initialize(net, inputs)
        epoch_start = 1
        print("Network initialized")

    error = losses.SpikeLoss().to(glv.rank)  # the loss is not defined here
    if multigpu:
        net = DDP(net, device_ids=[glv.rank], output_device=glv.rank)

    optim_type = params['Network']['optimizer']
    assert(optim_type in ['SGD', 'Adam', 'AdamW'])
    if optim_type == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=params['Network']['lr'])
    elif optim_type == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=params['Network']['lr'], betas=(0.9, 0.999))
    elif optim_type == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=params['Network']['lr'], betas=(0.9, 0.999))

    best_acc = 0

    l_states = learningStats()

    log_dir = f"{params['Network']['log_path']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}/"
    writer = SummaryWriter(log_dir)
    confu_mats = []
    for path in ['logs']:
        if not os.path.isdir(path):
            os.mkdir(path)
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.split(config_path)[-1]))

    (net.module if multigpu else net).train()
    for epoch in range(epoch_start, params['Network']['epochs'] + epoch_start):
        if multigpu:
            train_loader.sampler.set_epoch(epoch)
        l_states.training.reset()
        train_acc, loss = train(net, train_loader, optimizer, epoch, l_states, error)
        l_states.training.update()
        l_states.testing.reset()
        test_acc, confu_mat = test(net, test_loader, epoch, l_states, log_dir)
        l_states.testing.update()

        confu_mats.append(confu_mat)
        if glv.rank == 0:
            writer.add_scalars('Accuracy', {'train': train_acc,
                                            'test': test_acc}, epoch)
            writer.add_scalars('Loss', {'loss': loss}, epoch)
        np.save(log_dir + 'confusion_matrix.npy', np.stack(confu_mats))
