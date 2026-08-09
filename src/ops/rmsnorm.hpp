#pragma once

#include <stdexcept>

#include "cpu/kernels/rmsnorm.hpp"
#include "tensor/tensor.hpp"

namespace inference::ops {
    namespace impl {
        inline void rmsnorm_cpu(const Tensor& input, const Tensor& weight, Tensor& output, const std::size_t row_count,
                                const std::size_t hidden_size, const float epsilon) {
            switch (input.dtype()) {
            case types::DType::BF16:
                cpu::kernels::rmsnorm(input.data<cpu::bf16_t>(), weight.data<cpu::bf16_t>(), output.data<cpu::bf16_t>(), row_count, hidden_size,
                                      epsilon);
                return;
            default:
                throw std::runtime_error("no rmsnorm kernel for this dtype");
            }
        }
    } // namespace impl

    inline void rmsnorm(const Tensor& input, const Tensor& weight, Tensor& output, const std::size_t row_count, const std::size_t hidden_size,
                        const float epsilon) {
        switch (input.device()) {
        case types::Device::CPU:
            impl::rmsnorm_cpu(input, weight, output, row_count, hidden_size, epsilon);
            return;
        case types::Device::CUDA:
            throw std::runtime_error("cuda not implemented");
        }
    }
} // namespace inference::ops
