#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <cassert>

namespace {

template <typename scalar_t>
__global__ void neuron_forward_cuda_kernel(
    const scalar_t *__restrict__ in_I,
    scalar_t *__restrict__ delta_u,
    scalar_t *__restrict__ delta_u_t,
    scalar_t *__restrict__ outputs,
    const float theta_m, const float theta_s, const float theta_grad, const float threshold,
    const float is_forward_leaky, const float is_grad_exp,
    size_t neuron_num, size_t tot_size) {
    
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    float syn_m = 0, syn_s = 0, syn_grad = 0, u_last = 0, u = 0, out = 0;
    if (index < neuron_num) {
        for (; index < tot_size; index += neuron_num) {
            syn_m = (syn_m + in_I[index]) * (1 - theta_m);
            syn_s = (syn_s + in_I[index]) * (1 - theta_s);
            syn_grad = (syn_grad + in_I[index]) * (1 - theta_grad);
            
            if (!is_forward_leaky) {
                delta_u_t[index] = syn_grad;
                u = u_last + delta_u_t[index];
                delta_u[index] = delta_u_t[index];
            } else {
                u = (syn_m - syn_s) * theta_s / (theta_s - theta_m);
                delta_u[index] = u - u_last;
                delta_u_t[index] = is_grad_exp ? syn_grad : delta_u[index];
            }

            out = u >= threshold;
            u_last = out ? 0 : u;
            syn_m = out ? 0 : syn_m;
            syn_s = out ? 0 : syn_s;
            syn_grad = out ? 0 : syn_grad;

            outputs[index] = out;
        }
    }
}

template <typename scalar_t>
__global__ void neuron_backward_cuda_kernel(
    const scalar_t *__restrict__ grad_delta,
    const scalar_t *__restrict__ outputs,
    const scalar_t *__restrict__ delta_u,
    const scalar_t *__restrict__ delta_u_t,
    const scalar_t *__restrict__ syn_a,
    const scalar_t *__restrict__ partial_a,
    scalar_t *__restrict__ grad_in_,
    scalar_t *__restrict__ grad_w_,
    const float max_dudt_inv,
    size_t neuron_num, size_t tot_size) {
    
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    float partial_u = 0, partial_u_t = 0, partial_u_grad_w = 0, partial_u_grad_t = 0;
    int delta_t = 0;
    bool spiked = false, out = false;
    if (index < neuron_num) {
        for (index = tot_size - neuron_num + index; index >= 0; index -= neuron_num) {
            out = outputs[index] > 0;
            spiked |= out;

            partial_u = min(max(-1.0f / delta_u[index], -4.0f), 0.0f);
            partial_u_t = min(max(-1.0f / delta_u_t[index], -max_dudt_inv), 0.0f);
            partial_u_grad_w = out ? grad_delta[index] * partial_u : partial_u_grad_w;
            partial_u_grad_t = out ? grad_delta[index] * partial_u_t : partial_u_grad_t;

            delta_t = out ? 0 : delta_t + 1;
            grad_in_[index] = spiked ? partial_u_grad_t * partial_a[delta_t] : 0;
            grad_w_[index] = spiked ? partial_u_grad_w * syn_a[delta_t] : 0;
        }
    }
}

} // namespace

std::vector<torch::Tensor> neuron_forward_cuda(
    const torch::Tensor &in_I,
    const float theta_m,
    const float theta_s,
    const float theta_grad,
    const float threshold,
    const float is_forward_leaky,
    const float is_grad_exp) {

    auto delta_u = torch::zeros_like(in_I);
    auto delta_u_t = torch::zeros_like(in_I);
    auto outputs = torch::zeros_like(in_I);

    const auto tot_size = in_I.numel(), neuron_num = tot_size / in_I.size(0);

    const int threads = 1024;
    const auto blocks = (neuron_num + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(in_I.type(), "neuron_forward_cuda_kernel", ([&] {
                                   neuron_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
                                       in_I.data<scalar_t>(),
                                       delta_u.data<scalar_t>(),
                                       delta_u_t.data<scalar_t>(),
                                       outputs.data<scalar_t>(),
                                       theta_m, theta_s, theta_grad, threshold, is_forward_leaky, is_grad_exp,
                                       neuron_num, tot_size);
                               }));

    return {delta_u, delta_u_t, outputs};
}


std::vector<torch::Tensor> neuron_backward_cuda(
    const torch::Tensor &grad_delta,
    const torch::Tensor &outputs,
    const torch::Tensor &delta_u,
    const torch::Tensor &delta_u_t,
    const torch::Tensor &syn_a,
    const torch::Tensor &partial_a,
    const float max_dudt_inv) {

    auto grad_in_ = torch::zeros_like(outputs);
    auto grad_w_ = torch::zeros_like(outputs);

    const auto tot_size = outputs.numel(), neuron_num = tot_size / outputs.size(0);

    const int threads = 1024;
    const auto blocks = (neuron_num + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(grad_delta.type(), "neuron_backward_cuda_kernel", ([&] {
                                   neuron_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
                                       grad_delta.data<scalar_t>(),
                                       outputs.data<scalar_t>(),
                                       delta_u.data<scalar_t>(),
                                       delta_u_t.data<scalar_t>(),
                                       syn_a.data<scalar_t>(),
                                       partial_a.data<scalar_t>(),
                                       grad_in_.data<scalar_t>(),
                                       grad_w_.data<scalar_t>(),
                                       max_dudt_inv,
                                       neuron_num, tot_size);
                               }));

    return {grad_in_, grad_w_};
}
