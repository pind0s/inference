#pragma once
#include "backend/cpu/avx.hpp"
#include "tensor/tensor_view.hpp"

namespace inference::cpu::kernels {
    inline void add(TensorView<const bf16_t> lhs, TensorView<const bf16_t> rhs, TensorView<bf16_t> output) {
        const auto num_elems = output.size();
        constexpr std::size_t lanes = 16;
        const std::size_t simd_end = num_elems - num_elems % 16;
        for (std::size_t i = 0; i < simd_end; i += lanes) {
            const auto lhs_vec = avx::load_bf16_as_f32(&lhs[i]);
            const auto rhs_vec = avx::load_bf16_as_f32(&rhs[i]);
            avx::store_f32_as_bf16(&output[i], lhs_vec + rhs_vec);
        }

        for (std::size_t i = simd_end; i < num_elems; ++i) {
            output[i] = lhs[i].to_float() + rhs[i].to_float();
        }
    }
} // namespace inference::cpu::kernels
