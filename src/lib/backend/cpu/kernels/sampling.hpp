#pragma once
#include "backend/cpu/avx.hpp"
#include "tensor/tensor_view.hpp"
#include "types/token.hpp"
#include <algorithm>
#include <limits>

namespace inference::cpu::kernels {

    inline types::TokenId argmax_avx512bf16(const TensorView<const bf16_t> logits) {
        const auto num_elems = logits.size();
        float max_value = std::numeric_limits<float>::lowest();
        std::size_t block_index = 0;

        constexpr auto lanes = avx::lanes_v<avx::f32x16>;
        const auto simd_end = num_elems - (num_elems % lanes);

        for (std::size_t i = 0; i < simd_end; i += lanes) {
            const auto values = avx::load_bf16_as_f32(&logits[i]);
            const auto block_max = avx::reduce_max(values);
            if (block_max > max_value) [[unlikely]] {
                max_value = block_max;
                block_index = i;
            }
        }

        std::size_t maximum_index = 0;
        if (simd_end != 0) {
            maximum_index = std::max_element(logits.data() + block_index, logits.data() + block_index + lanes) - logits.data();
        }

        for (std::size_t i = simd_end; i < num_elems; ++i) {
            const auto value = logits[i].to_float();
            if (value > max_value) {
                max_value = value;
                maximum_index = i;
            }
        }

        return static_cast<types::TokenId>(maximum_index);
    }

} // namespace inference::cpu::kernels
