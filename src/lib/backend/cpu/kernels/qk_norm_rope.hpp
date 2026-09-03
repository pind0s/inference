#pragma once
#include "backend/cpu/bf16.hpp"
#include "backend/util/rope_cache.hpp"
#include "tensor/tensor_view.hpp"
#include <algorithm>
#include <cmath>

namespace inference::cpu::kernels {
    inline void qk_norm_rope(TensorView<bf16_t> query, TensorView<bf16_t> key, TensorView<const bf16_t> q_norm, TensorView<const bf16_t> k_norm,
                             const RopeCacheView cache, const std::size_t position, const float epsilon) {
        const auto query_head_count = query.dim(0);
        const auto key_head_count = key.dim(0);
        const auto head_count = std::max(query_head_count, key_head_count);
        const auto head_size = query.dim(1);
        const auto half_size = head_size / 2;

        for (std::size_t head = 0; head < head_count; ++head) {
            float query_sum_of_squares = 0.0F;
            float key_sum_of_squares = 0.0F;

            for (std::size_t index = 0; index < head_size; ++index) {
                if (head < query_head_count) {
                    const auto value = query(head, index).to_float();
                    query_sum_of_squares += value * value;
                }
                if (head < key_head_count) {
                    const auto value = key(head, index).to_float();
                    key_sum_of_squares += value * value;
                }
            }

            const auto query_scale = 1.0F / std::sqrt(query_sum_of_squares / static_cast<float>(head_size) + epsilon);
            const auto key_scale = 1.0F / std::sqrt(key_sum_of_squares / static_cast<float>(head_size) + epsilon);

            for (std::size_t index = 0; index < half_size; ++index) {
                const auto cosine = cache.cosine(position, index);
                const auto sine = cache.sine(position, index);

                if (head < query_head_count) {
                    const auto x = query(head, index).to_float() * q_norm(index).to_float() * query_scale;
                    const auto y = query(head, half_size + index).to_float() * q_norm(half_size + index).to_float() * query_scale;
                    query(head, index) = x * cosine - y * sine;
                    query(head, half_size + index) = y * cosine + x * sine;
                }

                if (head < key_head_count) {
                    const auto x = key(head, index).to_float() * k_norm(index).to_float() * key_scale;
                    const auto y = key(head, half_size + index).to_float() * k_norm(half_size + index).to_float() * key_scale;
                    key(head, index) = x * cosine - y * sine;
                    key(head, half_size + index) = y * cosine + x * sine;
                }
            }
        }
    }
} // namespace inference::cpu::kernels
