import torch
import torch.nn as nn
import global_v as glv
from layers.functions import neuron_forward, neuron_backward, bn_forward, bn_backward
from torch.cuda.amp import custom_fwd, custom_bwd


class LinearLayer(nn.Linear):
    def __init__(self, network_config, config, name):
        # extract information for kernel and inChannels
        in_features = config['n_inputs']
        out_features = config['n_outputs']
        self.threshold = config['threshold'] if 'threshold' in config else None
        self.name = name
        self.type = config['type']
        # self.in_shape = in_shape
        # self.out_shape = [out_features, 1, 1]

        if type(in_features) == int:
            n_inputs = in_features
        else:
            raise Exception('inFeatures should not be more than 1 dimesnion. It was: {}'.format(in_features.shape))
        if type(out_features) == int:
            n_outputs = out_features
        else:
            raise Exception('outFeatures should not be more than 1 dimesnion. It was: {}'.format(out_features.shape))

        super(LinearLayer, self).__init__(n_inputs, n_outputs, bias=False)
        self.weight = torch.nn.Parameter(self.weight.cuda(), requires_grad=True)
        self.bn_weight = torch.nn.Parameter(torch.ones(out_features,1, device='cuda'))
        self.bn_bias = torch.nn.Parameter(torch.zeros(out_features,1, device='cuda'))

        print("linear")
        print(self.name)
        # print(self.in_shape)
        # print(self.out_shape)
        print(f'Shape of weight is {list(self.weight.shape)}')
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

    def forward(self, x, labels=None):
        if glv.init_flag:
            glv.init_flag = False
            x = self.initialize(x)
            glv.init_flag = True
            return x

        self.weight_clipper()
        ndim = len(x.shape)
        assert(ndim == 3 or ndim == 5)
        if ndim == 5:
            T, n_batch, C, H, W = x.shape
            x = x.view(T, n_batch, C * H * W)
        config_n = glv.network_config
        theta_m = 1 / config_n['tau_m']
        theta_s = 1 / config_n['tau_s']
        theta_grad = 1 / config_n['tau_grad'] if config_n['gradient_type'] == 'exponential' else -123456789  #instead of None
        threshold = self.threshold
        y = LinearFunc.apply(x, self.weight, self.bn_weight, self.bn_bias, (theta_m, theta_s, theta_grad, threshold), labels)
        return y

    def weight_clipper(self):
        w = self.weight.data
        w = w.clamp(-4, 4)
        self.weight.data = w


class LinearFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, weight, bn_weight, bn_bias, config, labels):
        #input.shape: T * n_batch * N_in
        inputs, mean, var, weight_ = bn_forward(inputs, weight, bn_weight, bn_bias)

        in_I = torch.matmul(inputs, weight_.t())

        T, n_batch, N = in_I.shape
        theta_m, theta_s, theta_grad, threshold = torch.tensor(config)
        assert (theta_m != theta_s)
        delta_u, delta_u_t, outputs = neuron_forward(in_I, config)

        if labels is not None:
            glv.outputs_raw = outputs.clone()
            i2 = torch.arange(n_batch)
            # Add supervisory signal when synaptic potential is increasing:
            is_inc = (delta_u[:, i2, labels] > 0.05).float()
            _, i1 = torch.max(is_inc * torch.arange(1, T+1, device=is_inc.device).unsqueeze(-1), dim=0)
            outputs[i1, i2, labels] = (delta_u[i1, i2, labels] != 0).to(outputs)

            # i1 = (torch.ones(n_batch) * -1).long()
            # delta_u[i1, i2, labels] = torch.maximum(delta_u[i1, i2, labels], theta_s.to(outputs))
            # delta_u_t[i1, i2, labels] = torch.maximum(delta_u_t[i1, i2, labels], theta_s.to(outputs))

        ctx.save_for_backward(delta_u, delta_u_t, inputs, outputs, weight, bn_weight, bn_bias, mean, var)
        ctx.is_out_layer = labels != None

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        # shape of grad_delta: T * n_batch * N_out
        (delta_u, delta_u_t, inputs, outputs, weight, bn_weight, bn_bias, mean, var) = ctx.saved_tensors
        grad_delta *= outputs
        # sum_next = grad_delta.sum().item()
        # print("Max of dLdt: ", abs(grad_delta).max().item())

        grad_in_, grad_w_ = neuron_backward(grad_delta, outputs, delta_u, delta_u_t)
        weight_ = (weight - mean) / torch.sqrt(var + 1e-5) * bn_weight + bn_bias

        grad_input = torch.matmul(grad_in_, weight_) * inputs
        grad_weight = torch.sum(torch.matmul(grad_w_.transpose(1,2), inputs), dim=0)

        grad_weight, grad_bn_w, grad_bn_b = bn_backward(grad_weight, weight, bn_weight, bn_bias, mean, var)

        # sum_last = grad_input.sum().item()
        # assert(ctx.is_out_layer or abs(sum_next - sum_last) < 1)
        norm_mul = glv.network_config['norm_grad']
        return grad_input * 0.9, grad_weight, grad_bn_w * norm_mul, grad_bn_b * norm_mul, None, None, None