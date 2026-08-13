#pragma once
#include "cpu/avx.hpp"
#include "cpu/bf16.hpp"

#include <cmath>

namespace inference::cpu::kernels {
    inline void add(const __bf16* __restrict lhs, const __bf16* __restrict rhs, __bf16* out, const std::size_t N) {
        const std::size_t vectorized = N - (N % 16);
        constexpr std::size_t lanes = 16;

        [[omp::directive(parallel loop)]] for (std::size_t index = 0; index < vectorized; index += lanes) {
            const auto lhs_bf16 = avx::load<avx::bf16x16>(&lhs[index]);
            const auto rhs_bf16 = avx::load<avx::bf16x16>(&rhs[index]);

            auto sum = avx::bf16_to_f32(lhs_bf16) + avx::bf16_to_f32(rhs_bf16);

            const auto result = avx::f32_to_bf16(sum);
            avx::store(&out[index], result);
        }

        for (std::size_t index = vectorized; index < N; ++index) {
            out[index] = static_cast<__bf16>(static_cast<float>(lhs[index]) + static_cast<float>(rhs[index]));
        }
    }

    inline void silu_multiply(const __bf16* __restrict gate, const __bf16* __restrict up, __bf16* __restrict output, const std::size_t element_count) {
        for (std::size_t index = 0; index < element_count; ++index) {
            const auto gate_value = bf16::to_float(gate[index]);
            const auto silu = gate_value / (1.0F + std::exp(-gate_value));
            output[index] = bf16::from_float(silu * bf16::to_float(up[index]));
        }
    }

    namespace reference {
        inline void add(const __bf16* __restrict lhs, const __bf16* __restrict rhs, __bf16* __restrict out, const std::size_t N) {
            for (std::size_t index = 0; index < N; ++index) {
                out[index] = bf16::from_float(bf16::to_float(lhs[index]) + bf16::to_float(rhs[index]));
            }
        }
    } // namespace reference


} // namespace inference::cpu::kernels
