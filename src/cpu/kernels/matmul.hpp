#pragma once
#include "cpu/bf16.hpp"

#include <algorithm>

namespace inference::cpu::kernels {
    // todo cache blocking, look into _mm512_maskz_loadu_epi16 for tail, also clean up code a lil
    // https://en.algorithmica.org/hpc/algorithms/matmul/
    inline void matmul_avx512bf16(const bf16_t* __restrict a, const bf16_t* __restrict b_transposed, bf16_t* __restrict output, const std::size_t M,
                                  const std::size_t N, const std::size_t K) noexcept {
        const auto tile_rows = (M + 3) / 4;
        const auto tile_columns = (N + 3) / 4;
        const auto tile_count = tile_rows * tile_columns;

        [[omp::directive(parallel loop)]] for (std::size_t tile_index = 0; tile_index < tile_count; ++tile_index) {
            const auto x = (tile_index / tile_columns) * 4;
            const auto y = (tile_index % tile_columns) * 4;
            const auto rows = std::min(4UZ, M - x);
            const auto columns = std::min(4UZ, N - y);
            __m512 accum[4][4];

            for (auto& accumulator_row : accum) {
                for (auto& accumulator : accumulator_row) {
                    accumulator = _mm512_setzero_ps();
                }
            }

            std::size_t p = 0;
            for (; p + 32 <= K; p += 32) {
                __m512bh av[4];
                __m512bh bv[4];

                for (std::size_t i = 0; i < rows; ++i) {
                    av[i] = (__m512bh)_mm512_loadu_si512(a + (x + i) * K + p);
                }

                for (std::size_t j = 0; j < columns; ++j) {
                    bv[j] = (__m512bh)_mm512_loadu_si512(b_transposed + (y + j) * K + p);
                }

                for (std::size_t i = 0; i < rows; ++i) {
                    for (std::size_t j = 0; j < columns; ++j) {
                        accum[i][j] = _mm512_dpbf16_ps(accum[i][j], av[i], bv[j]);
                    }
                }
            }

            for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < columns; ++j) {
                    float sum = _mm512_reduce_add_ps(accum[i][j]);
                    for (std::size_t tail = p; tail < K; ++tail) {
                        sum += a[(x + i) * K + tail].to_float() * b_transposed[(y + j) * K + tail].to_float();
                    }
                    output[(x + i) * N + (y + j)] = bf16_t::from_float(sum);
                }
            }
        }
    }

} // namespace inference::cpu::kernels
