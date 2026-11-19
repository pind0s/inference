#pragma once
#include "backend/cpu/bf16.hpp"
#include "tensor/tensor.hpp"

namespace test::reference {
    [[nodiscard]] inline inference::Tensor add(const inference::Tensor& lhs, const inference::Tensor& rhs) {
        using namespace inference;
        auto result = lhs.to_host();
        const auto rhs_host = rhs.to_host();
        auto output = result.view<cpu::bf16_t>();
        const auto rhs_values = rhs_host.view<cpu::bf16_t>();

        for (std::size_t index = 0; index < output.size(); ++index) {
            output[index] = output[index].to_float() + rhs_values[index].to_float();
        }
        return result;
    }
} // namespace test::reference
