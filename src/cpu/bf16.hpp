#pragma once
#include <immintrin.h>

namespace inference::cpu {
    struct bf16_t {
        __bf16 value;

        // todo maybe we should just use cast directly? need to look into it
        [[gnu::always_inline]] [[nodiscard]] static bf16_t from_float(const float f32) {
            bf16_t result{};
            result.value = _mm_cvtness_sbh(f32);
            return result;
        }

        // todo think about removing this, since its just a simple C cast return (float)(__A);
        [[gnu::always_inline]] [[nodiscard]] float to_float() const {
            return _mm_cvtsbh_ss(value);
        }
    };
} // namespace inference::cpu
