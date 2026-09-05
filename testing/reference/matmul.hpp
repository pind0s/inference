#pragma once
#include "backend/cpu/bf16.hpp"
#include "util/to_host.hpp"

namespace test::reference {
    [[nodiscard]] inline inference::Tensor matmul(const inference::Tensor& input, const inference::Tensor& weights) {
        using namespace inference;
        const auto input_host = to_host(input);
        const auto weights_host = to_host(weights);
        const auto lhs = input_host.view<cpu::bf16_t>();
        const auto rhs = weights_host.view<cpu::bf16_t>();
        const auto column_count = rhs.dim(0);
        const auto inner_size = rhs.dim(1);
        const auto row_count = lhs.size() / inner_size;
        auto& cpu = Backend::get_backend<types::Device::CPU>();
        auto result = Tensor::empty({ row_count, column_count }, types::DType::BF16, cpu);
        auto output = result.view<cpu::bf16_t>();

        for (std::size_t row = 0; row < row_count; ++row) {
            for (std::size_t column = 0; column < column_count; ++column) {
                float sum = 0.0F;
                for (std::size_t inner = 0; inner < inner_size; ++inner) {
                    sum += lhs(row, inner).to_float() * rhs(column, inner).to_float();
                }
                output(row, column) = sum;
            }
        }
        return result;
    }
} // namespace test::reference
