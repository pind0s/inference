#pragma once
#include "backend/cpu/avx.hpp"
#include "backend/util/rope_cache.hpp"
#include "tensor/tensor_view.hpp"
#include <algorithm>
#include <cmath>

namespace inference::cpu::kernels {
    namespace detail {
        inline void norm_rope_tile(bf16_t* values, const bf16_t* norm, const std::size_t half_size, const float scale, const avx::f32x16 cosine,
                                   const avx::f32x16 sine) {
            const auto x = avx::load_bf16_as_f32(values) * avx::load_bf16_as_f32(norm) * scale;
            const auto y = avx::load_bf16_as_f32(values + half_size) * avx::load_bf16_as_f32(norm + half_size) * scale;
            avx::store_f32_as_bf16(values, x * cosine - y * sine);
            avx::store_f32_as_bf16(values + half_size, y * cosine + x * sine);
        }
    } // namespace detail

    inline void qk_norm_rope(TensorView<bf16_t> query, TensorView<bf16_t> key, TensorView<const bf16_t> q_norm, TensorView<const bf16_t> k_norm,
                             const backend::RopeCacheView cache, const std::size_t position, const float epsilon) {
        const auto query_head_count = query.dim(0);
        const auto key_head_count = key.dim(0);
        const auto head_count = std::max(query_head_count, key_head_count);
        const auto head_size = query.dim(1);
        const auto half_size = head_size / 2;

        [[omp::directive(parallel loop)]] for (std::size_t head = 0; head < head_count; ++head) {
            avx::f32x16 query_accumulator{};
            avx::f32x16 key_accumulator{};
            constexpr auto dot_lanes = avx::lanes_v<avx::bf16x32>;
            for (std::size_t index = 0; index < head_size; index += dot_lanes) {
                if (head < query_head_count) {
                    const auto values = avx::load<avx::bf16x32>(&query(head, index));
                    query_accumulator = avx::dot_bf16(values, values, query_accumulator);
                }
                if (head < key_head_count) {
                    const auto values = avx::load<avx::bf16x32>(&key(head, index));
                    key_accumulator = avx::dot_bf16(values, values, key_accumulator);
                }
            }
            const auto query_sum_of_squares = avx::reduce_add(query_accumulator);
            const auto key_sum_of_squares = avx::reduce_add(key_accumulator);

            const auto query_scale = 1.0F / std::sqrt(query_sum_of_squares / static_cast<float>(head_size) + epsilon);
            const auto key_scale = 1.0F / std::sqrt(key_sum_of_squares / static_cast<float>(head_size) + epsilon);

            constexpr auto lanes = avx::lanes_v<avx::f32x16>;
            for (std::size_t index = 0; index < half_size; index += lanes) {
                const auto cosine = avx::load<avx::f32x16>(&cache.cosine(position, index));
                const auto sine = avx::load<avx::f32x16>(&cache.sine(position, index));

                if (head < query_head_count) {
                    detail::norm_rope_tile(&query(head, index), &q_norm(index), half_size, query_scale, cosine, sine);
                }

                if (head < key_head_count) {
                    detail::norm_rope_tile(&key(head, index), &k_norm(index), half_size, key_scale, cosine, sine);
                }
            }
        }
    }
} // namespace inference::cpu::kernels
