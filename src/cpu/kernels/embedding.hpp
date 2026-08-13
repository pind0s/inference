#pragma once
#include <cstddef>

namespace inference::cpu::kernels {
    inline void embedding(const std::size_t token_id, const __bf16* __restrict weights, __bf16* __restrict output, const std::size_t hidden_size) {
        for (std::size_t hidden = 0; hidden < hidden_size; ++hidden) {
            output[hidden] = weights[token_id * hidden_size + hidden];
        }
    }
} // namespace inference::cpu::kernels
