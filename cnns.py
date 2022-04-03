import torch
import torch.nn as nn
import layers.conv as conv
import layers.pooling as pooling
import layers.dropout as dropout
import layers.linear as linear
from functools import reduce
from math import *
import global_v as glv


class Network(nn.Module):
    def __init__(self, input_shape):
        super(Network, self).__init__()
        self.layers = []
        network_config, layers_config = glv.network_config, glv.layers_config
        parameters = []
        print("Network Structure:")
        for key in layers_config:
            c = layers_config[key]
            if c['type'] == 'conv':
                self.layers.append(conv.ConvLayer(network_config, c, key, input_shape))
                input_shape = self.layers[-1].out_shape
                parameters.append(self.layers[-1].get_parameters())
            elif c['type'] == 'linear':
                self.layers.append(linear.LinearLayer(network_config, c, key, input_shape))
                input_shape = self.layers[-1].out_shape
                parameters.append(self.layers[-1].get_parameters())
            elif c['type'] == 'pooling':
                self.layers.append(pooling.PoolLayer(network_config, c, key, input_shape))
                input_shape = self.layers[-1].out_shape
            elif c['type'] == 'dropout':
                self.layers.append(dropout.DropoutLayer(c, key))
            else:
                raise Exception('Undefined layer type. It is: {}'.format(c['type']))
        self.my_parameters = nn.ParameterList(parameters)
        print("-----------------------------------------")

    def forward(self, inputs, labels, epoch, is_train):
        assert(is_train or labels==None)
        # spikes = f.psp(spike_input, self.network_config)
        spikes = inputs
        
        for i, l in enumerate(self.layers):
            if l.type == "dropout":
                if is_train:
                    spikes = l(spikes)
            elif i == len(self.layers) - 1:
                assert(l.type == 'linear')
                spikes = l.forward(spikes, labels)
            else:
                spikes = l.forward(spikes)

        return spikes

    def get_parameters(self):
        return self.my_parameters

    def weight_clipper(self):
        for l in self.layers:
            l.weight_clipper()

    def train(self):
        for l in self.layers:
            l.train()

    def eval(self):
        for l in self.layers:
            l.eval()


def initialize(net, inputs):
    avg_spike_init = glv.network_config['avg_spike_init']
    spikes = inputs
    for i, layer in enumerate(net.layers):
        if layer.type in ["conv", "linear"]:
            low, high = 1, 500
            while high / low >= 1.01:
                mid = sqrt(high * low)
                layer.weight.data *= mid
                layer_output = layer.forward(spikes)
                layer.weight.data /= mid
                n_neuron = reduce(lambda x,y: x*y, layer_output.shape[1:])
                avg_spike = torch.sum(layer_output) / n_neuron
                if avg_spike > avg_spike_init:
                    high = mid
                else:
                    low = mid
            layer.weight.data *= mid
            print(f'avg number of spikes = {avg_spike}')
        spikes = layer(spikes)
