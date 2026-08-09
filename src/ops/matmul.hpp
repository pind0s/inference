#pragma once

#include <stdexcept>

#include "cpu/kernels/matmul.hpp"
#include "tensor/tensor.hpp"

namespace inference::ops {
    namespace impl {
        inline void matmul_cpu(const Tensor& input, const Tensor& weights, Tensor& output, const std::size_t M, const std::size_t N,
                               const std::size_t K) {
            switch (input.dtype()) {
            case types::DType::BF16:
                cpu::kernels::matmul_avx512bf16(input.data<cpu::bf16_t>(), weights.data<cpu::bf16_t>(), output.data<cpu::bf16_t>(), M, N, K);
                return;
            default:
                throw std::runtime_error("no matmul kernel for this dtype");
            }
        }
    } // namespace impl

    inline void matmul(const Tensor& input, const Tensor& weights, Tensor& output, const std::size_t M, const std::size_t N,
                       const std::size_t K) {
        switch (input.device()) {
        case types::Device::CPU:
            impl::matmul_cpu(input, weights, output, M, N, K);
            return;
        case types::Device::CUDA:
            throw std::runtime_error("cuda not implemented");
        }
    }
} // namespace inference::ops
