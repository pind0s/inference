#pragma once

#include <stdexcept>

#include "cpu/kernels/self_attention.hpp"
#include "tensor/tensor.hpp"

namespace inference::ops {
    namespace impl {
        inline void self_attention_cpu(const Tensor& query, const Tensor& key, const Tensor& value, Tensor& output, const std::size_t position,
                                       const std::size_t query_head_count, const std::size_t key_value_head_count, const std::size_t head_size) {
            switch (query.dtype()) {
            case types::DType::BF16:
                cpu::kernels::self_attention(query.data<cpu::bf16_t>(), key.data<cpu::bf16_t>(), value.data<cpu::bf16_t>(),
                                             output.data<cpu::bf16_t>(), position, query_head_count, key_value_head_count, head_size);
                return;
            default:
                throw std::runtime_error("no self_attention kernel for this dtype");
            }
        }
    } // namespace impl

    inline void self_attention(const Tensor& query, const Tensor& key, const Tensor& value, Tensor& output, const std::size_t position,
                               const std::size_t query_head_count, const std::size_t key_value_head_count, const std::size_t head_size) {
        switch (query.device()) {
        case types::Device::CPU:
            impl::self_attention_cpu(query, key, value, output, position, query_head_count, key_value_head_count, head_size);
            return;
        case types::Device::CUDA:
            throw std::runtime_error("cuda not implemented");
        }
    }
} // namespace inference::ops
