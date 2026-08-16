#pragma once
#include "backend/cpu/bf16.hpp"
#include "tensor/tensor_view.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace test::reference::cpu {
    inline void self_attention(inference::TensorView<const inference::cpu::bf16_t> query, inference::TensorView<const inference::cpu::bf16_t> key,
                               inference::TensorView<const inference::cpu::bf16_t> value, inference::TensorView<inference::cpu::bf16_t> output,
                               const std::size_t position) {
        const auto query_head_count = query.dim(0);
        const auto key_value_head_count = key.dim(1);
        const auto head_size = query.dim(1);
        const auto queries_per_key_value = query_head_count / key_value_head_count;
        const auto attention_scale = 1.0F / std::sqrt(static_cast<float>(head_size));
        std::vector<float> weights(position + 1);

        for (std::size_t query_head = 0; query_head < query_head_count; ++query_head) {
            const auto key_value_head = query_head / queries_per_key_value;

            for (std::size_t key_position = 0; key_position < weights.size(); ++key_position) {
                float score = 0.0F;
                for (std::size_t index = 0; index < head_size; ++index) {
                    score += query(query_head, index).to_float() * key(key_position, key_value_head, index).to_float();
                }
                weights[key_position] = score * attention_scale;
            }

            const auto maximum_score = *std::ranges::max_element(weights);
            for (auto& weight : weights) {
                weight = std::exp(weight - maximum_score);
            }

            const auto weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0F);
            for (auto& weight : weights) {
                weight /= weight_sum;
            }

            for (std::size_t index = 0; index < head_size; ++index) {
                float result = 0.0F;
                for (std::size_t key_position = 0; key_position < weights.size(); ++key_position) {
                    result += weights[key_position] * value(key_position, key_value_head, index).to_float();
                }
                output(query_head, index) = result;
            }
        }
    }
} // namespace test::reference::cpu
