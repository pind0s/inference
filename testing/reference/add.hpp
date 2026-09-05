#pragma once
#include "backend/cpu/bf16.hpp"
#include "util/to_host.hpp"

namespace test::reference {
    [[nodiscard]] inline inference::Tensor add(const inference::Tensor& lhs, const inference::Tensor& rhs) {
        using namespace inference;
        auto result = to_host(lhs);
        const auto rhs_host = to_host(rhs);
        auto output = result.view<cpu::bf16_t>();
        const auto rhs_values = rhs_host.view<cpu::bf16_t>();

        for (std::size_t index = 0; index < output.size(); ++index) {
            output[index] = output[index].to_float() + rhs_values[index].to_float();
        }
        return result;
    }
} // namespace test::reference
