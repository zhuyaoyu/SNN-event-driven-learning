import torch
import torch.nn as nn
import torch.nn.functional as f
from layers.functions import neuron_forward, neuron_backward, bn_forward, bn_backward
import global_v as glv
import torch.backends.cudnn as cudnn
from torch.utils.cpp_extension import load_inline, load
from torch.cuda.amp import custom_fwd, custom_bwd
from datetime import datetime

cpp_wrapper = load(name="cpp_wrapper", sources=["layers/cpp_wrapper.cpp"], verbose=True)


class ConvLayer(nn.Conv2d):
    def __init__(self, network_config, config, name, groups=1):
        self.name = name
        self.threshold = config['threshold'] if 'threshold' in config else None
        self.type = config['type']
        in_features = config['in_channels']
        out_features = config['out_channels']
        kernel_size = config['kernel_size']

        padding = config['padding'] if 'padding' in config else 0
        stride = config['stride'] if 'stride' in config else 1
        dilation = config['dilation'] if 'dilation' in config else 1

        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size)
        elif len(kernel_size) > 2:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernel_size.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride)
        elif len(stride) > 2:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding)
        elif len(padding) > 2:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation)
        elif len(dilation) > 2:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        super(ConvLayer, self).__init__(in_features, out_features, kernel, stride, padding, dilation, groups,
                                        bias=False)
        self.weight = torch.nn.Parameter(self.weight.cuda(), requires_grad=True)
        self.bn_weight = torch.nn.Parameter(torch.ones(out_features, 1, 1, 1, device='cuda'))
        self.bn_bias = torch.nn.Parameter(torch.zeros(out_features, 1, 1, 1, device='cuda'))

        print('conv')
        print(f'Shape of weight is {list(self.weight.shape)}')  # Cout * Cin * Hk * Wk
        print(f'stride = {self.stride}, padding = {self.padding}, dilation = {self.dilation}, groups = {self.groups}')
        print("-----------------------------------------")

    def initialize(self, spikes):
        avg_spike_init = glv.network_config['avg_spike_init']
        from math import sqrt
        T = spikes.shape[0]
        t_start = T * 2 // 3

        low, high = 0.1, 100
        while high / low >= 1.01:
            mid = sqrt(high * low)
            self.bn_weight.data *= mid
            outputs = self.forward(spikes)
            self.bn_weight.data /= mid
            n_neuron = outputs[0].numel()
            avg_spike = torch.sum(outputs[t_start:]) / n_neuron
            if avg_spike > avg_spike_init / T * (T - t_start) * 1.3:
                high = mid
            else:
                low = mid
        self.bn_weight.data *= mid
        print(f'Average spikes per neuron = {torch.sum(outputs) / n_neuron}')
        return self.forward(spikes)

    def forward(self, x):
        if glv.init_flag:
            glv.init_flag = False
            x = self.initialize(x)
            glv.init_flag = True
            return x

        self.weight_clipper()
        config_n = glv.network_config
        theta_m = 1 / config_n['tau_m']
        theta_s = 1 / config_n['tau_s']
        theta_grad = 1 / config_n['tau_grad'] if config_n[
                                                     'gradient_type'] == 'exponential' else -123456789  # instead of None
        threshold = self.threshold
        y = ConvFunc.apply(x, self.weight, self.bn_weight, self.bn_bias,
                           (self.bias, self.stride, self.padding, self.dilation, self.groups),
                           (theta_m, theta_s, theta_grad, threshold))
        return y

    def weight_clipper(self):
        w = self.weight.data
        w = w.clamp(-4, 4)
        self.weight.data = w


class ConvFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, weight, bn_weight, bn_bias, conv_config, neuron_config):
        # input.shape: T * n_batch * C_in * H_in * W_in
        bias, stride, padding, dilation, groups = conv_config
        T, n_batch, C, H, W = inputs.shape

        inputs, mean, var, weight_ = bn_forward(inputs, weight, bn_weight, bn_bias)

        in_I = f.conv2d(inputs.reshape(T * n_batch, C, H, W), weight_, bias, stride, padding, dilation, groups)
        _, C, H, W = in_I.shape
        in_I = in_I.reshape(T, n_batch, C, H, W)

        delta_u, delta_u_t, outputs = neuron_forward(in_I, neuron_config)

        ctx.save_for_backward(delta_u, delta_u_t, inputs, outputs, weight, bn_weight, bn_bias, mean, var)
        ctx.conv_config = conv_config

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        # shape of grad_delta: T * n_batch * C * H * W
        (delta_u, delta_u_t, inputs, outputs, weight, bn_weight, bn_bias, mean, var) = ctx.saved_tensors
        bias, stride, padding, dilation, groups = ctx.conv_config
        grad_delta *= outputs
        # sum_next = grad_delta.sum().item()
        # print("Max of dLdt: ", abs(grad_delta).max().item())

        grad_in_, grad_w_ = neuron_backward(grad_delta, outputs, delta_u, delta_u_t)
        weight_ = (weight - mean) / torch.sqrt(var + 1e-5) * bn_weight + bn_bias

        T, n_batch, C, H, W = grad_delta.shape
        inputs = inputs.reshape(T * n_batch, *inputs.shape[2:])
        grad_in_, grad_w_ = map(lambda x: x.reshape(T * n_batch, C, H, W), [grad_in_, grad_w_])
        grad_input = cpp_wrapper.cudnn_convolution_backward_input(inputs.shape, grad_in_.to(weight_), weight_, padding,
                                                                  stride, dilation, groups,
                                                                  cudnn.benchmark, cudnn.deterministic,
                                                                  cudnn.allow_tf32) * inputs
        grad_weight = cpp_wrapper.cudnn_convolution_backward_weight(weight.shape, grad_w_.to(inputs), inputs, padding,
                                                                    stride, dilation, groups,
                                                                    cudnn.benchmark, cudnn.deterministic,
                                                                    cudnn.allow_tf32)

        grad_weight, grad_bn_w, grad_bn_b = bn_backward(grad_weight, weight, bn_weight, bn_bias, mean, var)

        # sum_last = grad_input.sum().item()
        # print(f'sum_next = {sum_next}, sum_last = {sum_last}')
        # assert(abs(sum_next - sum_last) < 1)
        norm_mul = glv.network_config['norm_grad']
        return grad_input.reshape(T, n_batch, *inputs.shape[1:]) * 0.9, grad_weight, \
               grad_bn_w * norm_mul, grad_bn_b * norm_mul, None, None, None
