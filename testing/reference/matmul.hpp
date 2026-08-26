#pragma once
#include "backend/cpu/bf16.hpp"
#include "tensor/tensor_view.hpp"

namespace test::reference {
    inline void matmul(inference::TensorView<const inference::cpu::bf16_t> lhs, inference::TensorView<const inference::cpu::bf16_t> rhs,
                       inference::TensorView<inference::cpu::bf16_t> output) {
        const auto row_count = lhs.dim(0);
        const auto column_count = rhs.dim(0);
        const auto inner_size = lhs.dim(1);

        for (std::size_t row = 0; row < row_count; ++row) {
            for (std::size_t column = 0; column < column_count; ++column) {
                float sum = 0.0F;
                for (std::size_t inner = 0; inner < inner_size; ++inner) {
                    sum += lhs(row, inner).to_float() * rhs(column, inner).to_float();
                }
                output(row, column) = sum;
            }
        }
    }
} // namespace test::reference
