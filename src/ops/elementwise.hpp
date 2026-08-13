#pragma once
#include "cpu/kernels/elementwise.hpp"
#include "tensor/tensor.hpp"

#include <stdexcept>

namespace inference::ops {
    namespace impl {
        inline void add_cpu(const Tensor& left, const Tensor& right, Tensor& output) {
            switch (left.dtype()) {
            case types::DType::BF16:
                cpu::kernels::add(left.data<__bf16>(), right.data<__bf16>(), output.data<__bf16>(), output.num_elems());
                return;
            default:
                throw std::runtime_error("no add kernel for this dtype");
            }
        }

        inline void silu_multiply_cpu(const Tensor& gate, const Tensor& up, Tensor& output) {
            switch (gate.dtype()) {
            case types::DType::BF16:
                cpu::kernels::silu_multiply(gate.data<__bf16>(), up.data<__bf16>(), output.data<__bf16>(), output.num_elems());
                return;
            default:
                throw std::runtime_error("no silu_multiply kernel for this dtype");
            }
        }
    } // namespace impl

    inline void add(const Tensor& left, const Tensor& right, Tensor& output) {
        switch (left.device()) {
        case types::Device::CPU:
            impl::add_cpu(left, right, output);
            return;
        case types::Device::CUDA:
            throw std::runtime_error("cuda not implemented");
        }
    }

    inline void silu_multiply(const Tensor& gate, const Tensor& up, Tensor& output) {
        switch (gate.device()) {
        case types::Device::CPU:
            impl::silu_multiply_cpu(gate, up, output);
            return;
        case types::Device::CUDA:
            throw std::runtime_error("cuda not implemented");
        }
    }
} // namespace inference::ops
