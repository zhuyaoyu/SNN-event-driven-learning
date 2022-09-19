import torch
import torch.nn as nn
import torch.nn.functional as f
import global_v as glv
from torch.cuda.amp import custom_fwd, custom_bwd
from layers.functions import readConfig


class PoolLayer(nn.Module):
    def __init__(self, network_config, config, name):
        super(PoolLayer, self).__init__()
        self.name = name
        self.layer_config = config
        self.network_config = network_config
        self.type = config['type']
        kernel_size = config['kernel_size']

        self.kernel = readConfig(kernel_size, 'kernelSize')
        # self.in_shape = in_shape
        # self.out_shape = [in_shape[0], int(in_shape[1] / kernel[0]), int(in_shape[2] / kernel[1])]
        print('pooling')
        # print(self.in_shape)
        # print(self.out_shape)
        print("-----------------------------------------")

    def forward(self, x):
        pool_type = glv.network_config['pooling_type']
        assert(pool_type in ['avg', 'max', 'adjusted_avg'])
        T, n_batch, C, H, W = x.shape
        x = x.reshape(T * n_batch, C, H, W)
        if pool_type == 'avg':
            x = f.avg_pool2d(x, self.kernel)
        elif pool_type == 'max':
            x = f.max_pool2d(x, self.kernel)
        elif pool_type == 'adjusted_avg':
            x = PoolFunc.apply(x, self.kernel)
        x = x.reshape(T, n_batch, *x.shape[1:])
        return x

    def get_parameters(self):
        return self.weight

    def forward_pass(self, x, epoch):
        y1 = self.forward(x)
        return y1

    def weight_clipper(self):
        return

class PoolFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, kernel):
        outputs = f.avg_pool2d(inputs, kernel)
        ctx.save_for_backward(outputs, torch.tensor(inputs.shape), torch.tensor(kernel))
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        (outputs, input_shape, kernel) = ctx.saved_tensors
        kernel = kernel.tolist()
        outputs = 1 / outputs
        outputs[outputs > kernel[0] * kernel[1] + 1] = 0
        outputs /= kernel[0] * kernel[1]
        grad = f.interpolate(grad_delta * outputs, size=input_shape.tolist()[2:])
        return grad, None