# **Training Spiking Neural Networks with Event-driven Backpropagation**

This repository is the official implementation of *Training Spiking Neural Networks with Event-driven Backpropagation*.

# Requirements
- pytorch=1.10.0
- torchvision=0.11.0
- spikingjelly

# Training

## Before running

Modify the data path and network settings in the config files (in the networks folder).

## Run the code
```
$ CUDA_VISIBLE_DEVICES=0 python main.py -config networks/config_file.yaml
```