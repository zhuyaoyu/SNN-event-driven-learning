import torch
import torch.nn as nn
import torch.nn.functional as f
from layers.neuron import neuron_forward, neuron_backward
import global_v as glv
import torch.backends.cudnn as cudnn
from torch.utils.cpp_extension import load_inline
from torch.cuda.amp import custom_fwd, custom_bwd
from datetime import datetime

cpp_wrapper = load_inline(
        name='cpp_wrapper',
        cpp_sources='using namespace at;',
        functions=[
            'cudnn_convolution_backward',
            'cudnn_convolution_backward_input',
            'cudnn_convolution_backward_weight'
        ],
        with_cuda=True
)

class ConvLayer(nn.Conv2d):
    def __init__(self, network_config, config, name, in_shape, groups=1):
        self.name = name
        self.layer_config = config
        self.type = config['type']
        in_features = config['in_channels']
        out_features = config['out_channels']
        kernel_size = config['kernel_size']

        if 'padding' in config:
            padding = config['padding']
        else:
            padding = 0

        if 'stride' in config:
            stride = config['stride']
        else:
            stride = 1

        if 'dilation' in config:
            dilation = config['dilation']
        else:
            dilation = 1

        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1])
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernel_size.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride)
        elif len(stride) == 2:
            stride = (stride[0], stride[1])
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding)
        elif len(padding) == 2:
            padding = (padding[0], padding[1])
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1])
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        super(ConvLayer, self).__init__(in_features, out_features, kernel, stride, padding, dilation, groups,
                                        bias=False)
        self.weight = torch.nn.Parameter(self.weight.cuda(), requires_grad=True)

        self.in_shape = in_shape
        self.out_shape = [out_features, int((in_shape[1]+2*padding[0]-kernel[0])/stride[0]+1),
                          int((in_shape[2]+2*padding[1]-kernel[1])/stride[1]+1)]
        print(self.name)
        print(self.in_shape)
        print(self.out_shape)
        print(list(self.weight.shape))
        print(f'stride = {self.stride}, padding = {self.padding}, dilation = {self.dilation}, groups = {self.groups}')
        print("-----------------------------------------")

    def forward(self, x):
        config_n = glv.network_config
        theta_m = 1 / config_n['tau_m']
        theta_s = 1 / config_n['tau_s']
        theta_grad = 1 / config_n['tau_grad'] if config_n['gradient_type'] == 'exponential' else -123456789  #instead of None
        threshold = self.layer_config['threshold']
        y = ConvFunc.apply(x, self.weight, (self.bias, self.stride, self.padding, self.dilation, self.groups), (theta_m, theta_s, theta_grad, threshold))
        return y

    def get_parameters(self):
        return self.weight

    def forward_pass(self, x, epoch):
        y = self.forward(x)
        return y

    def weight_clipper(self):
        w = self.weight.data
        w = w.clamp(-4, 4)
        self.weight.data = w


class ConvFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, weight, conv_config, neuron_config):
        #input.shape: T * n_batch * C_in * H_in * W_in
        bias, stride, padding, dilation, groups = conv_config
        T, n_batch, C, H, W = inputs.shape
        in_I = f.conv2d(inputs.reshape(T*n_batch, C, H, W), weight, bias, stride, padding, dilation, groups)
        _, C, H, W = in_I.shape
        in_I = in_I.reshape(T, n_batch, C, H, W)

        delta_u, delta_u_t, outputs = neuron_forward(in_I, neuron_config)

        ctx.save_for_backward(delta_u, delta_u_t, inputs, outputs, weight.to(inputs))
        ctx.conv_config = conv_config

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        # shape of grad_delta: T * n_batch * C * H * W
        (delta_u, delta_u_t, inputs, outputs, weight) = ctx.saved_tensors
        bias, stride, padding, dilation, groups = ctx.conv_config
        grad_delta *= outputs
        # print("sum of dLdt: ", grad_delta.sum().item(), abs(grad_delta).max().item())

        grad_in_, grad_w_ = neuron_backward(grad_delta, outputs, delta_u, delta_u_t)

        T, n_batch, C, H, W = grad_delta.shape
        inputs = inputs.reshape(T * n_batch, *inputs.shape[2:])
        grad_in_, grad_w_ = map(lambda x: x.reshape(T * n_batch, C, H, W), [grad_in_, grad_w_])
        grad_input = cpp_wrapper.cudnn_convolution_backward_input(inputs.shape, grad_in_.to(weight), weight, padding, stride, dilation, groups,
                                 cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32) * inputs
        grad_weight = cpp_wrapper.cudnn_convolution_backward_weight(weight.shape, grad_w_.to(inputs), inputs, padding, stride, dilation, groups,
                          cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32)

        # print("sum of dLdt in last layer: ", grad_input.sum().item())
        # print()
        return grad_input.reshape(T, n_batch, *inputs.shape[1:]), grad_weight, None, None, None