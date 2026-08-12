#pragma once
#include "cpu/bf16.hpp"

#include <cmath>

namespace inference::cpu::kernels {
    inline void add(const bf16_t* __restrict lhs, const bf16_t* __restrict rhs, bf16_t* __restrict out, std::size_t N) {
        const std::size_t vectorized = N - (N % 16);
        [[omp::directive(parallel loop)]] for (std::size_t index = 0; index < vectorized; index += 16) {
            auto lhs_bf16 = (__m256bh)_mm256_loadu_epi16(lhs + index);
            auto rhs_bf16 = (__m256bh)_mm256_loadu_epi16(rhs + index);

            auto lhs_f32 = _mm512_cvtpbh_ps(lhs_bf16);
            auto rhs_f32 = _mm512_cvtpbh_ps(rhs_bf16);

            auto sum = _mm512_add_ps(lhs_f32, rhs_f32);

            _mm256_storeu_epi16(out + index, (__m256i)_mm512_cvtneps_pbh(sum));
        }

        for (std::size_t index = vectorized; index < N; ++index) {
            out[index] = bf16_t::from_float(lhs[index].to_float() + rhs[index].to_float());
        }
    }

    inline void silu_multiply(const bf16_t* __restrict gate, const bf16_t* __restrict up, bf16_t* __restrict output, const std::size_t element_count) {
        for (std::size_t index = 0; index < element_count; ++index) {
            const auto gate_value = gate[index].to_float();
            const auto silu = gate_value / (1.0F + std::exp(-gate_value));
            output[index] = bf16_t::from_float(silu * up[index].to_float());
        }
    }
} // namespace inference::cpu::kernels
