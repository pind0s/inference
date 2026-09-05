#pragma once
#include "backend/cpu/bf16.hpp"
#include "types/token.hpp"
#include "util/to_host.hpp"

namespace test::reference {
    [[nodiscard]] inline inference::types::TokenId argmax(const inference::Tensor& logits) {
        const auto host = to_host(logits);
        const auto values = host.view<inference::cpu::bf16_t>();
        std::size_t maximum_index = 0;
        for (std::size_t index = 1; index < values.size(); ++index) {
            if (values[index].to_float() > values[maximum_index].to_float()) {
                maximum_index = index;
            }
        }
        return static_cast<inference::types::TokenId>(maximum_index);
    }
} // namespace test::reference
