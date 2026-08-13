#pragma once
#include "cpu/bf16.hpp"

#include <algorithm>
#include <cmath>
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
} // namespace inference::cpu::kernels
