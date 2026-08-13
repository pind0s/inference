#pragma once
#include "cpu/kernels/embedding.hpp"
#include "tensor/tensor.hpp"

#include <stdexcept>

namespace inference::ops {
    namespace impl {
        inline void embedding_cpu(const std::size_t token_id, const Tensor& weights, Tensor& output) {
            switch (weights.dtype()) {
            case types::DType::BF16:
                cpu::kernels::embedding(token_id, weights.data<__bf16>(), output.data<__bf16>(), output.num_elems());
                return;
            default:
                throw std::runtime_error("no embedding kernel for this dtype");
            }
        }
    } // namespace impl

    inline void embedding(const std::size_t token_id, const Tensor& weights, Tensor& output) {
        switch (weights.device()) {
        case types::Device::CPU:
            impl::embedding_cpu(token_id, weights, output);
            return;
        case types::Device::CUDA:
            throw std::runtime_error("cuda not implemented");
        }
    }
} // namespace inference::ops
