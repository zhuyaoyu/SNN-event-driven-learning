import torch
import torch.nn.functional as f
import global_v as glv
from torch.cuda.amp import custom_fwd, custom_bwd


def psp(inputs):
    n_steps = glv.network_config['n_steps']
    tau_s = glv.network_config['tau_s']
    syns = torch.zeros_like(inputs).to(glv.rank)
    syn = torch.zeros(syns.shape[1:]).to(glv.rank)

    for t in range(n_steps):
        syn = syn * (1 - 1 / tau_s) + inputs[t, ...]
        syns[t, ...] = syn / tau_s
    return syns


class SpikeLoss(torch.nn.Module):
    """
    This class defines different spike based loss modules that can be used to optimize the SNN.
    """

    def __init__(self):
        super(SpikeLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def spike_count(self, outputs, target):
        delta = loss_count.apply(outputs, target)
        return 1 / 2 * torch.sum(delta ** 2)

    def spike_kernel(self, outputs, target):
        out = grad_sign.apply(outputs)
        delta = psp(out - target)
        return 1 / 2 * torch.sum(delta ** 2)

    def spike_TET(self, outputs, target):
        out = grad_sign.apply(outputs)
        return f.cross_entropy(out, target.unsqueeze(0).repeat(out.shape[0], 1))

    def spike_times(self, outputs, target):
        return loss_spike_time.apply(outputs, target)


class loss_count(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, outputs, target):
        desired_count = glv.network_config['desired_count']
        undesired_count = glv.network_config['undesired_count']
        T = outputs.shape[0]
        assert(T == glv.T)
        out_count = torch.sum(outputs, dim=0)

        delta = (out_count - target) / T
        mask = torch.ones_like(out_count)
        mask[target == undesired_count] = 0
        mask[delta < 0] = 0
        delta[mask == 1] = 0
        mask = torch.ones_like(out_count)
        mask[target == desired_count] = 0
        mask[delta > 0] = 0
        delta[mask == 1] = 0
        delta = delta.unsqueeze_(0).repeat(T, 1, 1)
        return delta

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        sign = -1 if glv.network_config['loss_reverse'] else 1
        return sign * grad, None


class grad_sign(torch.autograd.Function):  # a and u is the increment of each time steps
    @staticmethod
    @custom_fwd
    def forward(ctx, outputs):
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        sign = -1 if glv.network_config['loss_reverse'] else 1
        return sign * grad

class loss_spike_time(torch.autograd.Function):  # a and u is the increment of each time steps
    @staticmethod
    @custom_fwd
    def forward(ctx, outputs):
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        sign = -1 if glv.network_config['loss_reverse'] else 1
        return sign * grad