#pragma once

#include <cmath>

#include "cpu/bf16.hpp"
namespace inference::cpu::kernels {
    inline void add(const bf16_t* __restrict left, const bf16_t* __restrict right, bf16_t* __restrict output,
                    const std::size_t element_count) noexcept {
        for (std::size_t index = 0; index < element_count; ++index) {
            output[index] = bf16_t::from_float(left[index].to_float() + right[index].to_float());
        }
    }

    inline void silu_multiply(const bf16_t* __restrict gate, const bf16_t* __restrict up, bf16_t* __restrict output,
                              const std::size_t element_count) noexcept {
        for (std::size_t index = 0; index < element_count; ++index) {
            const auto gate_value = gate[index].to_float();
            const auto silu = gate_value / (1.0F + std::exp(-gate_value));
            output[index] = bf16_t::from_float(silu * up[index].to_float());
        }
    }
} // namespace inference::cpu::kernels
