#pragma once
#include "cpu/avx.hpp"
#include "cpu/bf16.hpp"
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <limits>
#include <vector>
namespace inference::cpu::kernels {
    inline void self_attention(const __bf16* __restrict query, const __bf16* __restrict key, const __bf16* __restrict value, __bf16* __restrict output,
                               const std::size_t position, const std::size_t query_head_count, const std::size_t key_value_head_count,
                               const std::size_t head_size) {
        const auto queries_per_key_value = query_head_count / key_value_head_count;
        const auto attention_scale = 1.0F / std::sqrt(static_cast<float>(head_size));

        std::vector<float> scores(position + 1);
        for (std::size_t query_head = 0; query_head < query_head_count; ++query_head) {
            const auto key_value_head = query_head / queries_per_key_value;
            const auto query_base = query_head * head_size;
            auto maximum_score = -std::numeric_limits<float>::infinity();

            for (std::size_t key_position = 0; key_position <= position; ++key_position) {
                const auto key_base = (key_position * key_value_head_count + key_value_head) * head_size;
                float score = 0.0F;
                for (std::size_t index = 0; index < head_size; ++index) {
                    score += bf16::to_float(query[query_base + index]) * bf16::to_float(key[key_base + index]);
                }
                scores[key_position] = score * attention_scale;
                maximum_score = std::max(maximum_score, scores[key_position]);
            }

            float score_sum = 0.0F;
            for (std::size_t key_position = 0; key_position <= position; ++key_position) {
                scores[key_position] = std::exp(scores[key_position] - maximum_score);
                score_sum += scores[key_position];
            }

            for (std::size_t index = 0; index < head_size; ++index) {
                float result = 0.0F;
                for (std::size_t key_position = 0; key_position <= position; ++key_position) {
                    const auto value_base = (key_position * key_value_head_count + key_value_head) * head_size;
                    result += scores[key_position] / score_sum * bf16::to_float(value[value_base + index]);
                }
                output[query_base + index] = bf16::from_float(result);
            }
        }
    }

    // basically a one to one translation of func above
    inline void self_attention_optimized(const __bf16* __restrict query, const __bf16* __restrict key, const __bf16* __restrict value,
                                         __bf16* __restrict output, const std::size_t position, const std::size_t num_query_heads,
                                         const std::size_t num_kv_heads, const std::size_t head_size) {

        constexpr std::size_t lanes = 16;
        constexpr std::size_t bf16_dot_lanes = 32;
        const auto queries_per_key_value = num_query_heads / num_kv_heads;
        const auto attention_scale = 1.0F / std::sqrt(static_cast<float>(head_size));

        [[omp::directive(parallel loop)]] for (std::size_t query_head = 0; query_head < num_query_heads; ++query_head) {
            std::vector<float> scores(position + 1);

            const auto key_value_head = query_head / queries_per_key_value;
            const auto query_base = query_head * head_size;
            auto maximum_score = -std::numeric_limits<float>::infinity();

            for (std::size_t key_position = 0; key_position <= position; ++key_position) {
                const auto key_base = (key_position * num_kv_heads + key_value_head) * head_size;
                std::size_t index = 0;
                auto score_accumulator = _mm512_setzero_ps();

                for (; index + bf16_dot_lanes <= head_size; index += bf16_dot_lanes) {
                    const auto q = avx::load<avx::bf16x32>(&query[query_base + index]);
                    const auto k = avx::load<avx::bf16x32>(&key[key_base + index]);
                    score_accumulator = _mm512_dpbf16_ps(score_accumulator, (__m512bh)q, (__m512bh)k);
                }

                for (; index + lanes <= head_size; index += lanes) {
                    const auto q = avx::load<avx::bf16x16>(&query[query_base + index]);
                    const auto k = avx::load<avx::bf16x16>(&key[key_base + index]);
                    const auto q_f32 = avx::bf16_to_f32(q);
                    const auto k_f32 = avx::bf16_to_f32(k);
                    score_accumulator = _mm512_fmadd_ps(q_f32, k_f32, score_accumulator);
                }

                auto score = _mm512_reduce_add_ps(score_accumulator);
                for (; index < head_size; ++index) {
                    score += bf16::to_float(query[query_base + index]) * bf16::to_float(key[key_base + index]);
                }

                scores[key_position] = score * attention_scale;
                maximum_score = std::max(maximum_score, scores[key_position]);
            }

            float score_sum = 0.0F;
            std::size_t key_position = 0;
            for (; key_position + lanes <= scores.size(); key_position += lanes) {
                auto scores_vec = avx::load<avx::f32x16>(&scores[key_position]) - maximum_score;

                for (std::size_t i = 0; i < lanes; ++i) {
                    scores_vec[i] = std::exp(scores_vec[i]);
                }

                avx::store(&scores[key_position], scores_vec);
                score_sum += _mm512_reduce_add_ps(scores_vec);
            }

            // tail
            for (; key_position < scores.size(); ++key_position) {
                scores[key_position] = std::exp(scores[key_position] - maximum_score);
                score_sum += scores[key_position];
            }

            std::size_t index = 0;
            for (; index + lanes <= head_size; index += lanes) {
                avx::f32x16 result{};
                for (std::size_t key_position = 0; key_position < scores.size(); ++key_position) {
                    const auto value_base = (key_position * num_kv_heads + key_value_head) * head_size;
                    const auto values = avx::load<avx::bf16x16>(&value[value_base + index]);
                    const auto weight = scores[key_position] / score_sum;
                    result += avx::bf16_to_f32(values) * weight;
                }
                avx::store(&output[query_base + index], avx::f32_to_bf16(result));
            }

            for (; index < head_size; ++index) {
                float result = 0.0F;
                for (std::size_t key_position = 0; key_position < scores.size(); ++key_position) {
                    const auto value_base = (key_position * num_kv_heads + key_value_head) * head_size;
                    result += scores[key_position] / score_sum * bf16::to_float(value[value_base + index]);
                }
                output[query_base + index] = bf16::from_float(result);
            }
        }
    }
} // namespace inference::cpu::kernels
