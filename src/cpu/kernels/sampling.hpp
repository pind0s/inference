#pragma once
#include "cpu/avx.hpp"
#include <algorithm>
#include <limits>

namespace inference::cpu::kernels {

    inline std::size_t argmax_avx512bf16(const __bf16* logits, std::size_t num_elems) {
        float max_value = std::numeric_limits<float>::lowest();
        auto max_vec = avx::broadcast<avx::f32x16>(max_value);
        std::size_t block_index = 0;

        constexpr std::size_t lanes = 16;
        const auto simd_end = num_elems - (num_elems % lanes);

        for (std::size_t i = 0; i < simd_end; i += lanes) {
            const auto values = avx::load_bf16_as_f32(&logits[i]);
            if (_mm512_cmp_ps_mask(values, max_vec, _CMP_GT_OQ)) [[unlikely]] {
                max_value = avx::reduce_max(values);
                max_vec = avx::broadcast<avx::f32x16>(max_value);
                block_index = i;
            }
        }

        if (simd_end < num_elems) {
            const auto* tail_max = std::max_element(logits + simd_end, logits + num_elems);
            if (static_cast<float>(*tail_max) > max_value) {
                return static_cast<std::size_t>(tail_max - logits);
            }
        }

        return std::max_element(logits + block_index, logits + block_index + lanes) - logits;
    }

} // namespace inference::cpu::kernels
