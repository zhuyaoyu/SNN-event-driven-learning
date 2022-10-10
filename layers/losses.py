import torch
import torch.nn.functional as f
import global_v as glv
from torch.cuda.amp import custom_fwd, custom_bwd
from math import sqrt


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

    def spike_count(self, output, target):
        delta = loss_count.apply(output, target)
        return 1 / 2 * torch.sum(delta ** 2)

    def spike_kernel(self, output, target):
        out = grad_sign.apply(output)
        delta = psp(out - target)
        return 1 / 2 * torch.sum(delta ** 2)

    def spike_TET(self, output, target):
        output = output.permute(1, 2, 0)
        out = grad_sign.apply(output)
        return f.cross_entropy(out, target.unsqueeze(-1).repeat(1, out.shape[-1]))


class loss_count(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, output, target):
        desired_count = glv.network_config['desired_count']
        undesired_count = glv.network_config['undesired_count']
        T = output.shape[0]
        out_count = torch.sum(output, dim=0)

        delta = (out_count - target) / T
        delta[(target == desired_count) & (delta > 0) | (target == undesired_count) & (delta < 0)] = 0
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
