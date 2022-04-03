import torch
import torch.nn as nn
import global_v as glv
from layers.functions import neuron_forward, neuron_backward
from torch.cuda.amp import custom_fwd, custom_bwd


class LinearLayer(nn.Linear):
    def __init__(self, network_config, config, name, in_shape):
        # extract information for kernel and inChannels
        in_features = config['n_inputs']
        out_features = config['n_outputs']
        self.layer_config = config
        self.name = name
        self.type = config['type']
        self.in_shape = in_shape
        self.out_shape = [out_features, 1, 1]
        self.in_spikes = None
        self.out_spikes = None

        if type(in_features) == int:
            n_inputs = in_features
        else:
            raise Exception('inFeatures should not be more than 1 dimesnion. It was: {}'.format(in_features.shape))
        if type(out_features) == int:
            n_outputs = out_features
        else:
            raise Exception('outFeatures should not be more than 1 dimesnion. It was: {}'.format(out_features.shape))

        super(LinearLayer, self).__init__(n_inputs, n_outputs, bias=False, device=torch.device(glv.rank))


        print("linear")
        print(self.name)
        print(self.in_shape)
        print(self.out_shape)
        print(list(self.weight.shape))
        print("-----------------------------------------")

    def forward(self, x, labels=None):
        ndim = len(x.shape)
        assert(ndim == 3 or ndim == 5)
        if ndim == 5:
            T, n_batch, C, H, W = x.shape
            x = x.view(T, n_batch, C * H * W)
        config_n = glv.network_config
        theta_m = 1 / config_n['tau_m']
        theta_s = 1 / config_n['tau_s']
        theta_grad = 1 / config_n['tau_grad'] if config_n['gradient_type'] == 'exponential' else None
        threshold = self.layer_config['threshold']
        y = LinearFunc.apply(x, self.weight, (theta_m, theta_s, theta_grad, threshold), labels)
        return y

    def forward_pass(self, x, epoch):
        y = self.forward(x)
        return y

    def get_parameters(self):
        return self.weight

    def weight_clipper(self):
        w = self.weight.data
        w = w.clamp(-4, 4)
        self.weight.data = w


class LinearFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, weight, config, labels):
        #input.shape: T * n_batch * N_in
        in_I = torch.matmul(inputs, weight.t())

        T, n_batch, N = in_I.shape
        theta_m, theta_s, theta_grad, threshold = config
        assert (theta_m != theta_s)

        u_last, syn_m, syn_s, syn_grad, delta_u, delta_u_t, outputs = neuron_forward(in_I, config)

        if labels is not None:
            glv.outputs_raw = outputs.clone()
            i1 = (torch.ones(n_batch) * -1).long()
            i2 = torch.arange(n_batch)

            # Add supervisory signal only when no spike:
            # unspiked = (torch.sum(outputs[:, i2, labels], dim=0) == 0)
            # i1, i2, labels = i1[unspiked], i2[unspiked], labels[unspiked]

            outputs[i1, i2, labels] = (delta_u[i1, i2, labels] != 0).to(outputs)
            delta_u[i1, i2, labels] = torch.maximum(delta_u[i1, i2, labels], torch.tensor(theta_s).to(outputs))
            delta_u_t[i1, i2, labels] = torch.maximum(delta_u_t[i1, i2, labels], torch.tensor(theta_s).to(outputs))

        ctx.save_for_backward(delta_u, delta_u_t, inputs, outputs, weight)

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        # shape of grad_delta: n_batch * N_out * T
        (delta_u, delta_u_t, inputs, outputs, weight) = ctx.saved_tensors
        grad_delta *= outputs
        # print("sum of dLdt: ", grad_delta.sum().item(), abs(grad_delta).max().item())

        grad_in_, grad_w_ = neuron_backward(grad_delta, outputs, delta_u, delta_u_t)

        grad_input = torch.matmul(grad_in_, weight) * inputs
        grad_weight = torch.sum(torch.matmul(grad_w_.transpose(1,2), inputs), dim=0)

        # print("sum of dLdt in last layer: ", grad_input.sum().item())
        # print()
        return grad_input, grad_weight, None, None, None