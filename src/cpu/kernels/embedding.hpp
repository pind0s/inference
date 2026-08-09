#pragma once

#include <cstdint>

#include "cpu/bf16.hpp"
namespace inference::cpu::kernels {
    inline void embedding(const std::int32_t* __restrict input_ids, const bf16_t* __restrict weights, bf16_t* __restrict output,
                          const std::size_t token_count, const std::size_t hidden_size) noexcept {
        for (std::size_t token = 0; token < token_count; ++token) {
            for (std::size_t hidden = 0; hidden < hidden_size; ++hidden) {
                output[token * hidden_size + hidden] = weights[input_ids[token] * hidden_size + hidden];
            }
        }
    }
} // namespace inference::cpu::kernels
