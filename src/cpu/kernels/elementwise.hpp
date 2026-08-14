#pragma once
#include "cpu/avx.hpp"
#include <cmath>

namespace inference::cpu::kernels {
    inline void add(const __bf16* lhs, const __bf16* rhs, __bf16* output, const std::size_t num_elems) {
        constexpr std::size_t lanes = 16;
        const std::size_t simd_end = num_elems - (num_elems % 16);
        for (std::size_t i = 0; i < simd_end; i += lanes) {
            const auto lhs_vec = avx::load_bf16_as_f32(&lhs[i]);
            const auto rhs_vec = avx::load_bf16_as_f32(&rhs[i]);
            avx::store_f32_as_bf16(&output[i], lhs_vec + rhs_vec);
        }

        for (std::size_t i = simd_end; i < num_elems; ++i) {
            output[i] = static_cast<__bf16>(static_cast<float>(lhs[i]) + static_cast<float>(rhs[i]));
        }
    }

    inline void silu_multiply(const __bf16* __restrict gate, const __bf16* __restrict up, __bf16* __restrict output, const std::size_t element_count) {
        for (std::size_t index = 0; index < element_count; ++index) {
            const auto gate_value = static_cast<float>(gate[index]);
            const auto silu = gate_value / (1.0F + std::exp(-gate_value));
            output[index] = static_cast<__bf16>(silu * static_cast<float>(up[index]));
        }
    }

    namespace reference {
        inline void add(const __bf16* __restrict lhs, const __bf16* __restrict rhs, __bf16* __restrict out, const std::size_t N) {
            for (std::size_t index = 0; index < N; ++index) {
                out[index] = static_cast<__bf16>(static_cast<float>(lhs[index]) + static_cast<float>(rhs[index]));
            }
        }
    } // namespace reference

} // namespace inference::cpu::kernels
