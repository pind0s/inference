#pragma once
#include "backend/cpu/bf16.hpp"
#include "tensor/tensor_view.hpp"
#include "types/token.hpp"

namespace inference::cpu::kernels {
    inline void embedding(const types::TokenId token_id, const TensorView<const bf16_t> weights, const TensorView<bf16_t> output) {
        const auto hidden = output.size();
        std::ranges::copy_n(weights.data() + token_id * hidden, hidden, output.begin());
    }
} // namespace inference::cpu::kernels
