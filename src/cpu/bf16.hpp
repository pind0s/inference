#pragma once
#include <immintrin.h>

namespace inference::cpu::bf16 {
    [[gnu::always_inline]] [[nodiscard]] static __bf16 from_float(const float f32) {
        return _mm_cvtness_sbh(f32);
    }

    [[gnu::always_inline]] [[nodiscard]] static float to_float(const __bf16 value) {
        return _mm_cvtsbh_ss(value);
    }
} // namespace inference::cpu::bf16
