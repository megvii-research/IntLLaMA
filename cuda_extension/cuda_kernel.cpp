#include <torch/all.h>
#include <torch/types.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

//======================== 4 bit ========================
at::Tensor quant_gemm_cuda(
    torch::Tensor qinput,
    torch::Tensor in_scales, // N, 1
    torch::Tensor qweight,
    torch::Tensor w_scales, // torch.float16
    torch::Tensor w_zeros, // torch.uint8
    torch::Tensor w_step_scales, 
    torch::Tensor w_qstep,
    const size_t groups);

at::Tensor quant_gemm(
    torch::Tensor qinput,
    torch::Tensor in_scales,
    torch::Tensor qweight,
    torch::Tensor w_scales,
    torch::Tensor w_zeros,
    torch::Tensor w_step_scales,
    torch::Tensor w_qstep, 
    const size_t groups)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(qinput));
    torch::Tensor output = quant_gemm_cuda(
            qinput,
            in_scales,
            qweight,
            w_scales,
            w_zeros,
            w_step_scales,
            w_qstep,
            groups);
    return output;
}

std::vector<torch::Tensor> quant_pertoken_cuda(torch::Tensor &input);
std::vector<torch::Tensor> quant_pertoken(at::Tensor inputs)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(inputs));
    return quant_pertoken_cuda(inputs);
}

//======================== pybind =======================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quant_gemm", &quant_gemm, "a fused gemm kernel for UniformQLoRA speedup");
    m.def("tokenwise_quant", &quant_pertoken, "per-token quantization");
}
