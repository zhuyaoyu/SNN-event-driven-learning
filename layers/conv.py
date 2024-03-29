import torch
import torch.nn as nn
import torch.nn.functional as f
from layers.functions import neuron_forward, neuron_backward, bn_forward, bn_backward, readConfig, initialize
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

        self.kernel = readConfig(kernel_size, 'kernelSize')
        self.stride = readConfig(stride, 'stride')
        self.padding = readConfig(padding, 'stride')
        self.dilation = readConfig(dilation, 'stride')

        super(ConvLayer, self).__init__(in_features, out_features, self.kernel, self.stride, self.padding,
                                        self.dilation, groups, bias=False)
        self.weight = torch.nn.Parameter(self.weight.cuda(), requires_grad=True)
        self.norm_weight = torch.nn.Parameter(torch.ones(out_features, 1, 1, 1, device='cuda'))
        self.norm_bias = torch.nn.Parameter(torch.zeros(out_features, 1, 1, 1, device='cuda'))

        print('conv')
        print(f'Shape of weight is {list(self.weight.shape)}')  # Cout * Cin * Hk * Wk
        print(f'stride = {self.stride}, padding = {self.padding}, dilation = {self.dilation}, groups = {self.groups}')
        print("-----------------------------------------")

    def forward(self, x):
        if glv.init_flag:
            glv.init_flag = False
            x = initialize(self, x)
            glv.init_flag = True
            return x

        # self.weight_clipper()
        config_n = glv.network_config
        theta_m = 1 / config_n['tau_m']
        theta_s = 1 / config_n['tau_s']
        theta_grad = 1 / config_n['tau_grad'] if config_n[
                                                     'gradient_type'] == 'exponential' else -123456789  # instead of None
        y = ConvFunc.apply(x, self.weight, self.norm_weight, self.norm_bias,
                           (self.bias, self.stride, self.padding, self.dilation, self.groups),
                           (theta_m, theta_s, theta_grad, self.threshold))
        return y

    def weight_clipper(self):
        w = self.weight.data
        w = w.clamp(-4, 4)
        self.weight.data = w


class ConvFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, weight, norm_weight, norm_bias, conv_config, neuron_config):
        # input.shape: T * n_batch * C_in * H_in * W_in
        bias, stride, padding, dilation, groups = conv_config
        T, n_batch, C, H, W = inputs.shape

        inputs, mean, var, weight_ = bn_forward(inputs, weight, norm_weight, norm_bias)

        in_I = f.conv2d(inputs.reshape(T * n_batch, C, H, W), weight_, bias, stride, padding, dilation, groups)
        _, C, H, W = in_I.shape
        in_I = in_I.reshape(T, n_batch, C, H, W)

        delta_u, delta_u_t, outputs = neuron_forward(in_I, neuron_config)

        ctx.save_for_backward(delta_u, delta_u_t, inputs, outputs, weight, norm_weight, norm_bias, mean, var)
        ctx.conv_config = conv_config

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        # shape of grad_delta: T * n_batch * C * H * W
        (delta_u, delta_u_t, inputs, outputs, weight, norm_weight, norm_bias, mean, var) = ctx.saved_tensors
        bias, stride, padding, dilation, groups = ctx.conv_config
        grad_delta *= outputs
        # sum_next = grad_delta.sum().item()
        # print("Max of dLdt: ", abs(grad_delta).max().item())

        grad_in_, grad_w_ = neuron_backward(grad_delta, outputs, delta_u, delta_u_t)
        weight_ = (weight - mean) / torch.sqrt(var + 1e-5) * norm_weight + norm_bias

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

        grad_weight, grad_bn_w, grad_bn_b = bn_backward(grad_weight, weight, norm_weight, norm_bias, mean, var)

        # sum_last = grad_input.sum().item()
        # print(f'sum_next = {sum_next}, sum_last = {sum_last}')
        # assert(abs(sum_next - sum_last) < 1)
        return grad_input.reshape(T, n_batch, *inputs.shape[1:]) * 0.85, grad_weight, grad_bn_w, grad_bn_b, None, None, None
