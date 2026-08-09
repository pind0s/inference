#pragma once
#include <stdexcept>

#include "cpu/kernels/elementwise.hpp"
#include "tensor/tensor.hpp"

namespace inference::ops {
    namespace impl {
        inline void add_cpu(const Tensor& left, const Tensor& right, Tensor& output, const std::size_t element_count) {
            switch (left.dtype()) {
            case types::DType::BF16:
                cpu::kernels::add(left.data<cpu::bf16_t>(), right.data<cpu::bf16_t>(), output.data<cpu::bf16_t>(), element_count);
                return;
            default:
                throw std::runtime_error("no add kernel for this dtype");
            }
        }

        inline void silu_multiply_cpu(const Tensor& gate, const Tensor& up, Tensor& output, const std::size_t element_count) {
            switch (gate.dtype()) {
            case types::DType::BF16:
                cpu::kernels::silu_multiply(gate.data<cpu::bf16_t>(), up.data<cpu::bf16_t>(), output.data<cpu::bf16_t>(), element_count);
                return;
            default:
                throw std::runtime_error("no silu_multiply kernel for this dtype");
            }
        }
    } // namespace impl

    inline void add(const Tensor& left, const Tensor& right, Tensor& output, const std::size_t element_count) {
        switch (left.device()) {
        case types::Device::CPU:
            impl::add_cpu(left, right, output, element_count);
            return;
        case types::Device::CUDA:
            throw std::runtime_error("cuda not implemented");
        }
    }

    inline void silu_multiply(const Tensor& gate, const Tensor& up, Tensor& output, const std::size_t element_count) {
        switch (gate.device()) {
        case types::Device::CPU:
            impl::silu_multiply_cpu(gate, up, output, element_count);
            return;
        case types::Device::CUDA:
            throw std::runtime_error("cuda not implemented");
        }
    }
} // namespace inference::ops
