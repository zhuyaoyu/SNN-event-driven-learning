import torch


n_steps = None
syn_a = None
tau_s = None
tau_m = None
delta_syn_a = None
rank = None
network_config = None
layers_config = None
time_use = None


def init(config_n, config_l):
    global T, syn_a, delta_syn_a, tau_s, tau_m, grad_rec, outputs_raw
    global rank, network_config, layers_config, time_use
    network_config, layers_config = config_n, config_l
    if 'loss_reverse' not in network_config.keys():
        network_config['loss_reverse'] = True
    print('Whether loss is reversed: ', network_config['loss_reverse'])
    if 'amp' not in network_config.keys():
        network_config['amp'] = False

    T, tau_s, tau_m, grad_type = (config_n[x] for x in ('n_steps', 'tau_s', 'tau_m', 'gradient_type'))
    assert(grad_type in ['original', 'nonnegative', 'exponential'])
    if 'max_dudt_inv' not in network_config:
        network_config['max_dudt_inv'] = 123456789
    if 'avg_spike_init' not in network_config:
        network_config['avg_spike_init'] = 1

    syn_a, delta_syn_a = (torch.zeros(T + 1, device=torch.device(rank)) for _ in range(2))
    theta_m, theta_s = 1 / tau_m, 1 / tau_s
    if grad_type == 'exponential':
        assert('tau_grad' in config_n)
        tau_grad = config_n['tau_grad']
        theta_grad = 1 / tau_grad

    for t in range(T):
        t1 = t + 1
        syn_a[..., t] = ((1 - theta_m) ** t1 - (1 - theta_s) ** t1) * theta_s / (theta_s - theta_m)
        if grad_type == 'exponential':
            delta_syn_a[t] = (1 - theta_grad) ** t1 * theta_grad
        else:
            f = lambda t: ((1 - theta_m) ** t - (1 - theta_s) ** t) * theta_s / (theta_s - theta_m)
            delta_syn_a[t] = f(t1) - f(t1 - 1)
    if grad_type == 'nonnegative':
        delta_syn_a = torch.maximum(delta_syn_a, torch.zeros_like(delta_syn_a))
    print(syn_a, delta_syn_a)