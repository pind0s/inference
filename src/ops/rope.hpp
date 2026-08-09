#pragma once

#include <stdexcept>

#include "cpu/kernels/rope.hpp"
#include "tensor/tensor.hpp"

namespace inference::ops {
    namespace impl {
        inline void rope_cpu(Tensor& values, const std::size_t sequence_size, const std::size_t head_count, const std::size_t head_size,
                             const float theta) {
            switch (values.dtype()) {
            case types::DType::BF16:
                cpu::kernels::rope(values.data<cpu::bf16_t>(), sequence_size, head_count, head_size, theta);
                return;
            default:
                throw std::runtime_error("no rope kernel for this dtype");
            }
        }
    } // namespace impl

    inline void rope(Tensor& values, const std::size_t sequence_size, const std::size_t head_count, const std::size_t head_size,
                     const float theta) {
        switch (values.device()) {
        case types::Device::CPU:
            impl::rope_cpu(values, sequence_size, head_count, head_size, theta);
            return;
        case types::Device::CUDA:
            throw std::runtime_error("cuda not implemented");
        }
    }
} // namespace inference::ops
