#pragma once

#include <stdexcept>

#include "cpu/kernels/embedding.hpp"
#include "tensor/tensor.hpp"

namespace inference::ops {
    namespace impl {
        inline void embedding_cpu(const Tensor& input_ids, const Tensor& weights, Tensor& output, const std::size_t token_count,
                                  const std::size_t hidden_size) {
            switch (weights.dtype()) {
            case types::DType::BF16:
                cpu::kernels::embedding(input_ids.data<std::int32_t>(), weights.data<cpu::bf16_t>(), output.data<cpu::bf16_t>(), token_count,
                                        hidden_size);
                return;
            default:
                throw std::runtime_error("no embedding kernel for this dtype");
            }
        }
    } // namespace impl

    inline void embedding(const Tensor& input_ids, const Tensor& weights, Tensor& output, const std::size_t token_count,
                          const std::size_t hidden_size) {
        switch (weights.device()) {
        case types::Device::CPU:
            impl::embedding_cpu(input_ids, weights, output, token_count, hidden_size);
            return;
        case types::Device::CUDA:
            throw std::runtime_error("cuda not implemented");
        }
    }
} // namespace inference::ops
