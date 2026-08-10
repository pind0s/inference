#pragma once

#include <cstddef>

#include "cpu/bf16.hpp"
namespace inference::cpu::kernels {
    inline void embedding(const std::size_t token_id, const bf16_t* __restrict weights, bf16_t* __restrict output,
                          const std::size_t hidden_size) noexcept {
        for (std::size_t hidden = 0; hidden < hidden_size; ++hidden) {
            output[hidden] = weights[token_id * hidden_size + hidden];
        }
    }
} // namespace inference::cpu::kernels
