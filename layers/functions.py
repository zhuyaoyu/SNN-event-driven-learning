import torch
import global_v as glv

def neuron_forward(in_I, neuron_config):
    theta_m, theta_s, theta_grad, threshold = neuron_config
    assert (theta_m != theta_s)

    # syn_m & syn_s: (1-theta_m)^t & (1-theta_s)^t in eps(t)
    # syn_grad: (1-theta_grad)^t in backward
    u_last, syn_m, syn_s, syn_grad = (torch.zeros_like(in_I[0,...]) for _ in range(4))
    delta_u, delta_u_t, outputs = (torch.zeros_like(in_I) for _ in range(3))
    T = in_I.shape[0]
    for t in range(T):
        syn_m = (syn_m + in_I[t, ...]) * (1 - theta_m)
        syn_s = (syn_s + in_I[t, ...]) * (1 - theta_s)

        u = (syn_m - syn_s) * theta_s / (theta_s - theta_m)
        if glv.network_config['gradient_type'] == 'exponential':
            syn_grad = (syn_grad + in_I[t, ...]) * (1 - theta_grad)
            delta_u_t[t, ...] = syn_grad * theta_grad
        delta_u[t, ...] = u - u_last

        out = (u >= threshold).to(u)
        u_last = u * (1 - out)

        syn_m = syn_m * (1 - out)
        syn_s = syn_s * (1 - out)
        outputs[t, ...] = out

    delta_u_t = delta_u_t if glv.network_config['gradient_type'] == 'exponential' else delta_u.clone()

    return u_last, syn_m, syn_s, syn_grad, delta_u, delta_u_t, outputs


def neuron_backward(grad_delta, outputs, delta_u, delta_u_t):
    T = grad_delta.shape[0]
    max_dudt_inv = glv.network_config['max_dudt_inv']

    syn_a, partial_a = (x.to(outputs) for x in (glv.syn_a, -glv.delta_syn_a))
    grad_in_, grad_w_ = (torch.zeros_like(outputs) for _ in range(2))
    partial_u_grad_w, partial_u_grad_t = (torch.zeros_like(outputs[0,...]) for _ in range(2))
    delta_t = torch.zeros(outputs.shape[1:], device=outputs.device, dtype=torch.long)
    spiked = torch.zeros(outputs.shape[1:], device=outputs.device, dtype=torch.bool)

    for t in range(T - 1, -1, -1):
        out = outputs[t, ...]
        spiked |= out.bool()

        partial_u = torch.clamp(-1 / delta_u[t, ...], -4, 0) * out
        partial_u_t = torch.clamp(-1 / delta_u_t[t, ...], -max_dudt_inv, 0) * out
        # current time is t_m
        partial_u_grad_w = partial_u_grad_w * (1 - out) + grad_delta[t, ...] * partial_u
        partial_u_grad_t = partial_u_grad_t * (1 - out) + grad_delta[t, ...] * partial_u_t

        delta_t = (delta_t + 1) * (1 - out).long()
        grad_in_[t, ...] = partial_u_grad_t * partial_a[delta_t] * spiked.to(partial_a)
        grad_w_[t, ...] = partial_u_grad_w * syn_a[delta_t] * spiked.to(syn_a)

    return grad_in_, grad_w_