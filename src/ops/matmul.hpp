#pragma once

#include <stdexcept>

#include "cpu/kernels/matmul.hpp"
#include "tensor/tensor.hpp"

namespace inference::ops {
    namespace impl {
        inline void matmul_cpu(const Tensor& input, const Tensor& weights, Tensor& output) {
            // todo
            const auto rows = input.num_elems() / weights.dim(1);
            switch (input.dtype()) {
            case types::DType::BF16:
                cpu::kernels::matmul_avx512bf16(input.data<cpu::bf16_t>(), weights.data<cpu::bf16_t>(), output.data<cpu::bf16_t>(), rows,
                                                weights.dim(0), weights.dim(1));
                return;
            default:
                throw std::runtime_error("no matmul kernel for this dtype");
            }
        }
    } // namespace impl

    inline void matmul(const Tensor& input, const Tensor& weights, Tensor& output) {
        switch (input.device()) {
        case types::Device::CPU:
            impl::matmul_cpu(input, weights, output);
            return;
        case types::Device::CUDA:
            throw std::runtime_error("cuda not implemented");
        }
    }
} // namespace inference::ops
