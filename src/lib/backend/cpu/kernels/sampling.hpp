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

        constexpr std::size_t lanes = 16;
        const auto simd_end = num_elems - (num_elems % lanes);

        for (std::size_t i = 0; i < simd_end; i += lanes) {
            const auto values = avx::load_bf16_as_f32(&logits[i]);
            const auto block_max = avx::reduce_max(values);
            if (block_max > max_value) [[unlikely]] {
                max_value = block_max;
                block_index = i;
            }
        }

        if (simd_end < num_elems) {
            const auto* tail_max = std::max_element(logits.data() + simd_end, logits.data() + num_elems);
            if (tail_max->to_float() > max_value) {
                return static_cast<types::TokenId>(tail_max - logits.data());
            }
        }

        return static_cast<types::TokenId>(std::max_element(logits.data() + block_index, logits.data() + block_index + lanes) - logits.data());
    }

} // namespace inference::cpu::kernels
