import torch
import global_v as glv
from torch.utils.cpp_extension import load

try:
    neuron_cuda = load(name="neuron_cuda", sources=["layers/neuron_cuda.cpp", 'layers/neuron_cuda_kernel.cu'],
                       verbose=True)
except:
    print('Cannot load cuda neuron kernel.')


def readConfig(data, name):
    if type(data) == int:
        res = (data, data)
    else: # str
        try:
            assert(data[0] == '(' and data[-1] == ')')
            data = data[1:len(data)-1]
            x, y = map(int, data.split(','))
            res = (x, y)
        except:
            raise Exception(f'The format of {name} is illegal!')
    return res


def initialize(layer, spikes):
    avg_spike_init = glv.network_config['avg_spike_init']
    from math import sqrt
    T = spikes.shape[0]
    t_start = T * 2 // 3

    low, high = 0.1, 100
    while high / low >= 1.01:
        mid = sqrt(high * low)
        layer.bn_weight.data *= mid
        outputs = layer.forward(spikes)
        layer.bn_weight.data /= mid
        n_neuron = outputs[0].numel()
        avg_spike = torch.sum(outputs[t_start:]) / n_neuron
        if avg_spike > avg_spike_init / T * (T - t_start) * 1.3:
            high = mid
        else:
            low = mid
    layer.threshold.data /= mid
    print(f'Average spikes per neuron = {torch.sum(outputs) / n_neuron}')
    return layer.forward(spikes)


def norm(inputs):
    T = inputs.shape[0]
    t_start = T * 2 // 3
    if (inputs >= 0).all():
        num_spike = (torch.sum(inputs[t_start:]) + 1e-5)
        target_spike = inputs.numel() / T * (T - t_start) / T
        inputs = inputs / num_spike * target_spike
    return inputs


def bn_forward(inputs, weight, bn_weight, bn_bias):
    # inputs = norm(inputs)
    C = weight.shape[0]
    # print(weight.shape)
    mean, var = torch.mean(weight.reshape(C, -1), dim=1), torch.std(weight.reshape(C, -1), dim=1) ** 2
    shape = (-1, 1, 1, 1) if len(weight.shape) == 4 else (-1, 1)
    mean, var, bn_weight, bn_bias = [x.reshape(*shape) for x in [mean, var, bn_weight, bn_bias]]
    weight_ = (weight - mean) / torch.sqrt(var + 1e-5) * bn_weight + bn_bias
    return inputs, mean, var, weight_


def bn_backward(grad_weight, weight, bn_weight, bn_bias, mean, var):
    C = weight.shape[0]
    std_inv = 1 / torch.sqrt(var + 1e-5)
    shape = (-1, 1, 1, 1) if len(weight.shape) == 4 else (-1, 1)
    weight_ = (weight - mean) * std_inv * bn_weight.reshape(*shape) + bn_bias.reshape(*shape)
    grad_bn_b = torch.sum(grad_weight.reshape(C, -1), dim=1).reshape(bn_bias.shape)
    grad_bn_w = torch.sum((grad_weight * weight_).reshape(C, -1), dim=1).reshape(bn_weight.shape)
    grad_weight *= bn_weight.reshape(*shape)
    m = weight.numel() // C
    grad_var = grad_weight * (weight - mean) / m * (-0.5) * std_inv ** 3
    grad_mean = -grad_weight * std_inv
    grad_weight = grad_weight * std_inv + grad_var * 2 * (weight - mean) / m + grad_mean / m
    return grad_weight, grad_bn_w, grad_bn_b


@torch.jit.script
def neuron_forward_py(in_I, theta_m, theta_s, theta_grad, threshold, is_forward_leaky, is_grad_exp):
    # syn_m & syn_s: (1-theta_m)^t & (1-theta_s)^t in eps(t)
    # syn_grad: (1-theta_grad)^t in backward
    u_last = torch.zeros_like(in_I[0])
    syn_m, syn_s, syn_grad = torch.zeros_like(in_I[0]), torch.zeros_like(in_I[0]), torch.zeros_like(in_I[0])
    delta_u, delta_u_t, outputs = torch.zeros_like(in_I), torch.zeros_like(in_I), torch.zeros_like(in_I)
    T = in_I.shape[0]
    for t in range(T):
        syn_m = (syn_m + in_I[t]) * (1 - theta_m)
        syn_s = (syn_s + in_I[t]) * (1 - theta_s)
        syn_grad = (syn_grad + in_I[t]) * (1 - theta_grad)

        if not is_forward_leaky:
            delta_u_t[t] = syn_grad
            u = u_last + delta_u_t[t]
            delta_u[t] = delta_u_t[t]
        else:
            u = (syn_m - syn_s) * theta_s / (theta_s - theta_m)
            delta_u[t] = u - u_last
            delta_u_t[t] = syn_grad if is_grad_exp else delta_u[t]

        out = (u >= threshold).to(u)
        u_last = u * (1 - out)

        syn_m = syn_m * (1 - out)
        syn_s = syn_s * (1 - out)
        syn_grad = syn_grad * (1 - out)
        outputs[t] = out

    return delta_u, delta_u_t, outputs


@torch.jit.script
def neuron_backward_py(grad_delta, outputs, delta_u, delta_u_t, syn_a, partial_a, max_dudt_inv):
    T = grad_delta.shape[0]

    grad_in_, grad_w_ = torch.zeros_like(outputs), torch.zeros_like(outputs)
    partial_u_grad_w, partial_u_grad_t = torch.zeros_like(outputs[0]), torch.zeros_like(outputs[0])
    delta_t = torch.zeros(outputs.shape[1:], device=outputs.device, dtype=torch.long)
    spiked = torch.zeros_like(outputs[0])

    for t in range(T - 1, -1, -1):
        out = outputs[t]
        spiked += (1 - spiked) * out

        partial_u = torch.clamp(-1 / delta_u[t], -4, 0)
        partial_u_t = torch.clamp(-1 / delta_u_t[t], -max_dudt_inv, 0)
        # current time is t_m
        partial_u_grad_w = partial_u_grad_w * (1 - out) + grad_delta[t] * partial_u * out
        partial_u_grad_t = partial_u_grad_t * (1 - out) + grad_delta[t] * partial_u_t * out

        delta_t = (delta_t + 1) * (1 - out).long()
        grad_in_[t] = partial_u_grad_t * partial_a[delta_t] * spiked.to(partial_a)
        grad_w_[t] = partial_u_grad_w * syn_a[delta_t] * spiked.to(syn_a)

    return grad_in_, grad_w_


def neuron_forward(in_I, neuron_config):
    theta_m, theta_s, theta_grad, threshold = torch.tensor(neuron_config).to(in_I)
    assert (theta_m != theta_s)
    is_grad_exp = torch.tensor(glv.network_config['gradient_type'] == 'exponential')
    is_forward_leaky = torch.tensor(glv.network_config['forward_type'] == 'leaky')
    if glv.network_config['backend'] == 'python':
        return neuron_forward_py(in_I, theta_m, theta_s, theta_grad, threshold, is_forward_leaky, is_grad_exp)
    elif glv.network_config['backend'] == 'cuda':
        # global neuron_cuda
        # if neuron_cuda is None:
        theta_m, theta_s, theta_grad, threshold = neuron_config
        return neuron_cuda.forward(in_I, theta_m, theta_s, theta_grad, threshold, is_forward_leaky, is_grad_exp)
    else:
        raise Exception('Unrecognized computation backend.')


def neuron_backward(grad_delta, outputs, delta_u, delta_u_t):
    syn_a, partial_a = glv.syn_a.to(outputs), -glv.delta_syn_a.to(outputs)
    max_dudt_inv = torch.tensor(glv.network_config['max_dudt_inv'])
    if glv.network_config['backend'] == 'python':
        return neuron_backward_py(grad_delta, outputs, delta_u, delta_u_t, syn_a, partial_a, max_dudt_inv)
    elif glv.network_config['backend'] == 'cuda':
        max_dudt_inv = max_dudt_inv.item()
        return neuron_cuda.backward(grad_delta, outputs, delta_u, delta_u_t, syn_a, partial_a, max_dudt_inv)
    else:
        raise Exception('Unrecognized computation backend.')


if __name__ == '__main__':
    T = 12
    glv.rank = 0
    config = dict()
    config['gradient_type'] = 'exponential'
    config['forward_type'] = 'nonleaky'
    for key, val in zip(('n_steps', 'tau_s', 'tau_m', 'tau_grad', 'threshold'), (T, 7, 4, 3.5, 1)):
        config[key] = val
    glv.init(config, config)
    neuron_cuda = load(name="neuron_cuda", sources=["neuron_cuda.cpp", 'neuron_cuda_kernel.cu'], verbose=True)
    shape = (T, 50, 3, 32, 32)

    neuron_config = [1 / glv.network_config[key] for key in ('tau_m', 'tau_s', 'tau_grad')] + [
        glv.network_config['threshold']]
    in_I = torch.rand(*shape, device=torch.device('cuda'))
    glv.network_config['backend'] = 'python'
    delta_u_py, delta_u_t_py, outputs_py = neuron_forward(in_I, neuron_config)
    glv.network_config['backend'] = 'cuda'
    delta_u_cuda, delta_u_t_cuda, outputs_cuda = neuron_forward(in_I, neuron_config)
    print(torch.sum(delta_u_py), torch.sum(delta_u_cuda))
    assert (torch.sum(torch.abs(delta_u_py - delta_u_cuda)).item() <= 1e-3)
    assert (torch.sum(torch.abs(delta_u_t_py - delta_u_t_cuda)).item() <= 1e-3)
    assert (torch.sum(torch.abs(outputs_py - outputs_cuda)) <= 1e-3)

    grad_delta = torch.rand(*shape, device=torch.device('cuda'))
    outputs = torch.round(torch.rand_like(grad_delta))
    delta_u = torch.rand_like(grad_delta) * 8 - 4
    delta_u_t = torch.rand_like(grad_delta) * 8 - 4
    glv.network_config['backend'] = 'python'
    grad_in_py, grad_w_py = neuron_backward(grad_delta, outputs, delta_u, delta_u_t)
    glv.network_config['backend'] = 'cuda'
    grad_in_cuda, grad_w_cuda = neuron_backward(grad_delta, outputs, delta_u, delta_u_t)
    print(torch.sum(grad_in_py), torch.sum(grad_in_cuda))
    assert (torch.sum(torch.abs(grad_in_py - grad_in_cuda)) <= 1e-3)
    assert (torch.sum(torch.abs(grad_w_py - grad_w_cuda)) <= 1e-3)
