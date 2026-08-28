#pragma once
#include "tensor/tensor_view.hpp"
#include <cmath>

namespace inference::cpu::kernels {
    inline void silu(TensorView<const bf16_t> gate, TensorView<const bf16_t> up, TensorView<bf16_t> output) {
        const auto element_count = output.size();
        for (std::size_t index = 0; index < element_count; ++index) {
            const auto gate_value = gate[index].to_float();
            const auto silu = gate_value / (1.0F + std::exp(-gate_value));
            output[index] = silu * up[index].to_float();
        }
    }
} // namespace inference::cpu::kernels
