import torch
import global_v as glv
from torch.utils.cpp_extension import load

try:
    neuron_cuda = load(name="neuron_cuda", sources=["layers/neuron_cuda.cpp", 'layers/neuron_cuda_kernel.cu'],
                       verbose=True)
except:
    print('Cannot use cuda backend for neuron functions.')


@torch.jit.script
def neuron_forward_py(in_I, theta_m, theta_s, theta_grad, threshold, is_grad_exp):
    # syn_m & syn_s: (1-theta_m)^t & (1-theta_s)^t in eps(t)
    # syn_grad: (1-theta_grad)^t in backward
    u_last, syn_m, syn_s, syn_grad = torch.zeros_like(in_I[0]), torch.zeros_like(in_I[0]), torch.zeros_like(
        in_I[0]), torch.zeros_like(in_I[0])
    delta_u, delta_u_t, outputs = torch.zeros_like(in_I), torch.zeros_like(in_I), torch.zeros_like(in_I)
    T = in_I.shape[0]
    for t in range(T):
        syn_m = (syn_m + in_I[t]) * (1 - theta_m)
        syn_s = (syn_s + in_I[t]) * (1 - theta_s)

        u = (syn_m - syn_s) * theta_s / (theta_s - theta_m)
        delta_u[t] = u - u_last
        if is_grad_exp:
            syn_grad = (syn_grad + in_I[t]) * (1 - theta_grad)
            delta_u_t[t] = syn_grad * theta_grad

        out = (u >= threshold).to(u)
        u_last = u * (1 - out)

        syn_m = syn_m * (1 - out)
        syn_s = syn_s * (1 - out)
        outputs[t] = out

    delta_u_t = delta_u_t if is_grad_exp else delta_u.clone()

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
    if glv.network_config['backend'] == 'python':
        return neuron_forward_py(in_I, theta_m, theta_s, theta_grad, threshold, is_grad_exp)
    elif glv.network_config['backend'] == 'cuda':
        theta_m, theta_s, theta_grad, threshold = (x.item() for x in (theta_m, theta_s, theta_grad, threshold))
        return neuron_cuda.forward(in_I, theta_m, theta_s, theta_grad, threshold, is_grad_exp)
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
    for key, val in zip(('n_steps', 'tau_s', 'tau_m', 'tau_grad', 'threshold'), (T, 7, 4, 3.5, 1)):
        config[key] = val
    glv.init(config, config)
    neuron_cuda = load(name="neuron_cuda", sources=["neuron_cuda.cpp", 'neuron_cuda_kernel.cu'], verbose=True)
    shape = (T, 50, 3, 32, 32)

    neuron_config = [1 / glv.network_config[key] for key in ('tau_m', 'tau_s', 'tau_grad')] + [glv.network_config['threshold']]
    in_I = torch.rand(*shape, device=torch.device('cuda'))
    glv.network_config['backend'] = 'python'
    delta_u_py, delta_u_t_py, outputs_py = neuron_forward(in_I, neuron_config)
    glv.network_config['backend'] = 'cuda'
    delta_u_cuda, delta_u_t_cuda, outputs_cuda = neuron_forward(in_I, neuron_config)
    print(torch.sum(delta_u_py), torch.sum(delta_u_cuda))
    assert(torch.sum(torch.abs(delta_u_py - delta_u_cuda)).item() <= 1e-3)
    assert(torch.sum(torch.abs(delta_u_t_py - delta_u_t_cuda)).item() <= 1e-3)
    assert(torch.sum(torch.abs(outputs_py - outputs_cuda)) <= 1e-3)

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
