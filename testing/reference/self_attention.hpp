#pragma once
#include "backend/cpu/bf16.hpp"
#include "util/to_host.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace test::reference {
    [[nodiscard]] inline inference::Tensor self_attention(const inference::Tensor& query_tensor, const inference::Tensor& key_tensor,
                                                          const inference::Tensor& value_tensor, const std::size_t position) {
        using namespace inference;
        auto result = to_host(query_tensor);
        const auto key_host = to_host(key_tensor);
        const auto value_host = to_host(value_tensor);
        const auto key = key_host.view<cpu::bf16_t>();
        const auto value = value_host.view<cpu::bf16_t>();
        auto output = result.view<cpu::bf16_t>();
        const auto query_head_count = output.dim(0);
        const auto key_value_head_count = key.dim(1);
        const auto head_size = output.dim(1);
        const auto queries_per_key_value = query_head_count / key_value_head_count;
        const auto attention_scale = 1.0F / std::sqrt(static_cast<float>(head_size));
        std::vector<float> weights(position + 1);

        for (std::size_t query_head = 0; query_head < query_head_count; ++query_head) {
            const auto key_value_head = query_head / queries_per_key_value;

            for (std::size_t key_position = 0; key_position < weights.size(); ++key_position) {
                float score = 0.0F;
                for (std::size_t index = 0; index < head_size; ++index) {
                    score += output(query_head, index).to_float() * key(key_position, key_value_head, index).to_float();
                }
                weights[key_position] = score * attention_scale;
            }

            const auto maximum_score = *std::ranges::max_element(weights);
            float weight_sum = 0.0F;
            for (auto& weight : weights) {
                weight = std::exp(weight - maximum_score);
                weight_sum += weight;
            }

            for (auto& weight : weights) {
                weight /= weight_sum;
            }

            for (std::size_t index = 0; index < head_size; ++index) {
                float sum = 0.0F;
                for (std::size_t key_position = 0; key_position < weights.size(); ++key_position) {
                    sum += weights[key_position] * value(key_position, key_value_head, index).to_float();
                }
                output(query_head, index) = sum;
            }
        }
        return result;
    }
} // namespace test::reference
