#pragma once
#include "cpu/avx.hpp"

#include <cmath>

namespace inference::cpu::kernels {
    inline void add(const __bf16* __restrict lhs, const __bf16* __restrict rhs, __bf16* out, const std::size_t N) {
        const std::size_t vectorized = N - (N % 16);
        constexpr std::size_t lanes = 16;

        [[omp::directive(parallel loop)]] for (std::size_t index = 0; index < vectorized; index += lanes) {
            const auto lhs_f32 = avx::load_bf16_as_f32(&lhs[index]);
            const auto rhs_f32 = avx::load_bf16_as_f32(&rhs[index]);

            const auto sum = lhs_f32 + rhs_f32;

            avx::store_f32_as_bf16(&out[index], sum);
        }

        for (std::size_t index = vectorized; index < N; ++index) {
            out[index] = static_cast<__bf16>(static_cast<float>(lhs[index]) + static_cast<float>(rhs[index]));
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
