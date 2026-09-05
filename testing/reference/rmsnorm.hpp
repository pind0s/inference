#pragma once
#include "backend/cpu/bf16.hpp"
#include "util/to_host.hpp"
#include <cmath>
#include <span>

namespace test::reference {
    [[nodiscard]] inline inference::Tensor rmsnorm(const inference::Tensor& input_tensor, const inference::Tensor& weight_tensor, const float epsilon) {
        using namespace inference;
        auto result = to_host(input_tensor);
        const auto weight_host = to_host(weight_tensor);
        const auto weight = weight_host.view<cpu::bf16_t>();
        auto output = result.view<cpu::bf16_t>();
        const auto row_size = weight.size();
        const auto row_count = result.size() / row_size;

        for (std::size_t row_index = 0; row_index < row_count; ++row_index) {
            auto row = std::span{ output }.subspan(row_index * row_size, row_size);
            float sum_of_squares = 0.0f;

            for (const auto element : row) {
                const auto value = element.to_float();
                sum_of_squares += value * value;
            }

            const auto scale = 1.0f / std::sqrt(sum_of_squares / static_cast<float>(row_size) + epsilon);
            for (std::size_t index = 0; index < row_size; ++index) {
                row[index] = row[index].to_float() * weight[index].to_float() * scale;
            }
        }
        return result;
    }
} // namespace test::reference
