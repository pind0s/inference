#pragma once
#include "backend/cpu/avx.hpp"
#include "tensor/tensor_view.hpp"
#include <algorithm>

namespace inference::cpu::kernels {
    // todo cache blocking, look into a masked tail load, also clean up code a lil
    //  stuff like tile size and etc should be defined once and reused in the loop.
    // https://en.algorithmica.org/hpc/algorithms/matmul/
    inline void matmul_avx512bf16(TensorView<const bf16_t> a, TensorView<const bf16_t> b, TensorView<bf16_t> output) {
        const auto inner_size = b.dim(1);
        const auto columns = b.dim(0);
        const auto rows = a.size() / inner_size;
        const auto tile_rows = (rows + 3) / 4;
        const auto tile_columns = (columns + 3) / 4;
        const auto tile_count = tile_rows * tile_columns;

        [[omp::directive(parallel loop)]] for (std::size_t tile_index = 0; tile_index < tile_count; ++tile_index) {
            const auto x = (tile_index / tile_columns) * 4;
            const auto y = (tile_index % tile_columns) * 4;
            const auto rows_in_tile = std::min(4UZ, rows - x);
            const auto columns_in_tile = std::min(4UZ, columns - y);
            avx::f32x16 accum[4][4];

            for (auto& accumulator_row : accum) {
                for (auto& accumulator : accumulator_row) {
                    accumulator = avx::zero();
                }
            }

            std::size_t p = 0;
            for (; p + 32 <= inner_size; p += 32) {
                avx::bf16x32 av[4];
                avx::bf16x32 bv[4];

                for (std::size_t i = 0; i < rows_in_tile; ++i) {
                    av[i] = avx::load<avx::bf16x32>(&a[(x + i) * inner_size + p]);
                }

                for (std::size_t j = 0; j < columns_in_tile; ++j) {
                    bv[j] = avx::load<avx::bf16x32>(&b(y + j, p));
                }

                for (std::size_t i = 0; i < rows_in_tile; ++i) {
                    for (std::size_t j = 0; j < columns_in_tile; ++j) {
                        accum[i][j] = avx::dot_bf16(av[i], bv[j], accum[i][j]);
                    }
                }
            }

            for (std::size_t i = 0; i < rows_in_tile; ++i) {
                for (std::size_t j = 0; j < columns_in_tile; ++j) {
                    float sum = avx::reduce_add(accum[i][j]);
                    for (std::size_t tail = p; tail < inner_size; ++tail) {

                        sum += a(x + i, tail).to_float() * b(y + j, tail).to_float();
                    }
                    output[(x + i) * columns + (y + j)] = sum;
                }
            }
        }
    }
} // namespace inference::cpu::kernels
