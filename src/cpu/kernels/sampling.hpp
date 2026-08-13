#pragma once
#include "cpu/avx.hpp"
#include <algorithm>
#include <limits>

namespace inference::cpu::kernels {

    inline std::size_t argmax_avx(const __bf16* logits, std::size_t num_elems) {
        float max_value = std::numeric_limits<float>::lowest();
        auto current_max = avx::broadcast<avx::f32x16>(max_value);

        std::size_t block_index = 0;
        std::size_t i = 0;
        constexpr std::size_t lanes = 16;
        for (; i + lanes <= num_elems; i += lanes) {
            const auto values = avx::load_bf16_as_f32(&logits[i]);

            const auto larger = _mm512_cmp_ps_mask(values, current_max, _CMP_GT_OQ);
            if (larger != 0) [[unlikely]] {
                max_value = avx::reduce_max(values);
                current_max = avx::broadcast<avx::f32x16>(max_value);
                block_index = i;
            }
        }

        if (i < num_elems) {
            const auto* tail_max = std::max_element(logits + i, logits + num_elems);
            if (static_cast<float>(*tail_max) > max_value) {
                return static_cast<std::size_t>(tail_max - logits);
            }
        }

        return std::max_element(logits + block_index, logits + block_index + 16) - logits;
    }

} // namespace inference::cpu::kernels
