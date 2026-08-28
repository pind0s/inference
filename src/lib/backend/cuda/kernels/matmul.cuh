#pragma once
#include "tensor/tensor_view.hpp"
#include <cublas_v2.h>
#include <cuda_bf16.h>

namespace inference::gpu::kernels {

    inline void matmul_cublas_bf16(const cublasHandle_t handle, const TensorView<const __nv_bfloat16> input, const TensorView<const __nv_bfloat16> weights,
                                   const TensorView<__nv_bfloat16>& output) {
        const auto inner_size = weights.dim(1);
        const auto columns = weights.dim(0);
        const auto rows = input.size() / inner_size;
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(columns), static_cast<int>(rows), static_cast<int>(inner_size), &alpha,
                     weights.data(), CUDA_R_16BF, static_cast<int>(inner_size), input.data(), CUDA_R_16BF, static_cast<int>(inner_size), &beta,
                     output.data(), CUDA_R_16BF, static_cast<int>(columns), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
} // namespace inference::gpu::kernels
