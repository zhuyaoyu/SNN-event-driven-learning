#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> neuron_forward_cuda(
    const torch::Tensor &in_I,
    const float theta_m,
    const float theta_s,
    const float theta_grad,
    const float threshold,
    const float is_forward_leaky,
    const float is_grad_exp);

std::vector<torch::Tensor> neuron_backward_cuda(
    const torch::Tensor &grad_delta,
    const torch::Tensor &outputs,
    const torch::Tensor &delta_u,
    const torch::Tensor &delta_u_t,
    const torch::Tensor &syn_a,
    const torch::Tensor &partial_a,
    const float max_dudt_inv);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM((x).type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> neuron_forward(
    const torch::Tensor &in_I,
    const float theta_m,
    const float theta_s,
    const float theta_grad,
    const float threshold,
    const float is_forward_leaky,
    const float is_grad_exp) {

    CHECK_INPUT(in_I);
    return neuron_forward_cuda(in_I, theta_m, theta_s, theta_grad, threshold, is_forward_leaky, is_grad_exp);
}

std::vector<torch::Tensor> neuron_backward(
    const torch::Tensor &grad_delta,
    const torch::Tensor &outputs,
    const torch::Tensor &delta_u,
    const torch::Tensor &delta_u_t,
    const torch::Tensor &syn_a,
    const torch::Tensor &partial_a,
    const float max_dudt_inv) {

    CHECK_INPUT(grad_delta);
    CHECK_INPUT(outputs);
    CHECK_INPUT(delta_u);
    CHECK_INPUT(delta_u_t);
    CHECK_INPUT(syn_a);
    CHECK_INPUT(partial_a);
    return neuron_backward_cuda(grad_delta, outputs, delta_u, delta_u_t, syn_a, partial_a, max_dudt_inv);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &neuron_forward, "Neuron forward (CUDA)");
    m.def("backward", &neuron_backward, "Neuron backward (CUDA)");
}
