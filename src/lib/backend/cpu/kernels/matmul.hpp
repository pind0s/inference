#pragma once
#include "backend/cpu/avx.hpp"
#include "tensor/tensor_view.hpp"
#include <algorithm>

namespace inference::cpu::kernels {

    inline void gemv(TensorView<const bf16_t> a, TensorView<const bf16_t> b, TensorView<bf16_t> output) {
        const auto inner_size = a.size();
        const auto output_size = b.dim(0);

        constexpr std::size_t outputs_per_tile = 8;
        constexpr auto lanes = avx::lanes_v<avx::bf16x32>;
        const auto tile_count = (output_size + outputs_per_tile - 1) / outputs_per_tile;

        [[omp::directive(parallel loop)]] for (std::size_t tile = 0; tile < tile_count; ++tile) {
            const auto output_begin = tile * outputs_per_tile;
            const auto outputs_in_tile = std::min(outputs_per_tile, output_size - output_begin);
            avx::f32x16 accumulators[outputs_per_tile]{};

            for (std::size_t inner = 0; inner < inner_size; inner += lanes) {
                const auto a_tile = avx::load<avx::bf16x32>(&a[inner]);
                for (std::size_t output_index = 0; output_index < outputs_in_tile; ++output_index) {
                    const auto b_tile = avx::load<avx::bf16x32>(&b(output_begin + output_index, inner));
                    accumulators[output_index] = avx::dot_bf16(a_tile, b_tile, accumulators[output_index]);
                }
            }

            for (std::size_t output_index = 0; output_index < outputs_in_tile; ++output_index) {
                output[output_begin + output_index] = avx::reduce_add(accumulators[output_index]);
            }
        }
    }
} // namespace inference::cpu::kernels
