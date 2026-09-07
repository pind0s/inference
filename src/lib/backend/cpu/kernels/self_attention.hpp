#pragma once
#include "backend/cpu/avx.hpp"
#include "tensor/tensor_view.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

namespace inference::cpu::kernels {
    // todo this is a disaster and we can simplify this A LOT
    inline void self_attention(TensorView<const bf16_t> query, TensorView<const bf16_t> key, TensorView<const bf16_t> value, TensorView<bf16_t> output,
                               std::size_t pos) {

        constexpr auto lanes = avx::lanes_v<avx::f32x16>;
        constexpr auto bf16_dot_lanes = avx::lanes_v<avx::bf16x32>;
        const auto num_query_heads = query.dim(0);
        const auto num_kv_heads = key.dim(1);
        const auto head_size = query.dim(1);
        const auto queries_per_key_value = num_query_heads / num_kv_heads;
        const auto attention_scale = 1.0F / std::sqrt(static_cast<float>(head_size));

        [[omp::directive(parallel loop)]] for (std::size_t query_head = 0; query_head < num_query_heads; ++query_head) {
            // todo introduce flash attention
            std::vector<float> scores(pos + 1);

            const auto key_value_head = query_head / queries_per_key_value;
            auto maximum_score = std::numeric_limits<float>::lowest();

            for (std::size_t key_position = 0; key_position <= pos; ++key_position) {
                avx::f32x16 score_accumulator{};

                for (std::size_t index = 0; index < head_size; index += bf16_dot_lanes) {
                    const auto q = avx::load<avx::bf16x32>(&query(query_head, index));
                    const auto k = avx::load<avx::bf16x32>(&key(key_position, key_value_head, index));
                    score_accumulator = avx::dot_bf16(q, k, score_accumulator);
                }

                const auto score = avx::reduce_add(score_accumulator);

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
                score_sum += avx::reduce_add(scores_vec);
            }

            // tail
            for (; key_position < scores.size(); ++key_position) {
                scores[key_position] = std::exp(scores[key_position] - maximum_score);
                score_sum += scores[key_position];
            }

            for (std::size_t index = 0; index < head_size; index += lanes) {
                avx::f32x16 result{};
                for (std::size_t key_position = 0; key_position < scores.size(); ++key_position) {
                    const auto values = avx::load_bf16_as_f32(&value(key_position, key_value_head, index));
                    const auto weight = scores[key_position] / score_sum;
                    result += values * weight;
                }
                avx::store_f32_as_bf16(&output(query_head, index), result);
            }
        }
    }
} // namespace inference::cpu::kernels
