import torch


def init(config_n, config_l=None):
    global T, T_train, syn_a, delta_syn_a, tau_s, tau_m, grad_rec, outputs_raw
    global rank, network_config, layers_config, time_use, req_grad, init_flag
    init_flag = False

    network_config, layers_config = config_n, config_l

    if 'loss_reverse' not in network_config.keys():
        network_config['loss_reverse'] = True

    if 'encoding' not in network_config.keys():
        network_config['encoding'] = 'None'
    if 'amp' not in network_config.keys():
        network_config['amp'] = False
    if 'backend' not in network_config.keys():
        network_config['backend'] = 'python'
    if 'norm_grad' not in network_config.keys():
        network_config['norm_grad'] = 1

    if 'max_dudt_inv' not in network_config:
        network_config['max_dudt_inv'] = 123456789
    if 'avg_spike_init' not in network_config:
        network_config['avg_spike_init'] = 1
    if 'weight_decay' not in network_config:
        network_config['weight_decay'] = 0
    if 't_train' not in network_config:
        network_config['t_train'] = network_config['n_steps']

    T, tau_s, tau_m, grad_type = (config_n[x] for x in ('n_steps', 'tau_s', 'tau_m', 'gradient_type'))
    if 'forward_type' not in network_config:
        network_config['forward_type'] = 'leaky'

    assert(network_config['forward_type'] in ['leaky', 'nonleaky'])
    assert(grad_type in ['original', 'exponential'])
    assert(not (network_config['forward_type'] == 'nonleaky' and grad_type == 'original'))

    syn_a, delta_syn_a = (torch.zeros(T + 1, device=torch.device(rank)) for _ in range(2))
    theta_m, theta_s = 1 / tau_m, 1 / tau_s
    if grad_type == 'exponential':
        assert('tau_grad' in config_n)
        tau_grad = config_n['tau_grad']
        theta_grad = 1 / tau_grad

    for t in range(T):
        t1 = t + 1
        syn_a[t] = ((1 - theta_m) ** t1 - (1 - theta_s) ** t1) * theta_s / (theta_s - theta_m)
        if grad_type == 'exponential':
            delta_syn_a[t] = (1 - theta_grad) ** t1
        else:
            f = lambda t: ((1 - theta_m) ** t - (1 - theta_s) ** t) * theta_s / (theta_s - theta_m)
            delta_syn_a[t] = f(t1) - f(t1 - 1)
    # print(syn_a, delta_syn_a)