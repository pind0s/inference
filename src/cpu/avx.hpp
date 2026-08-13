#pragma once
#include <cstring>

namespace inference::cpu::avx {
    using f32x8 = float __attribute__((vector_size(8 * sizeof(float))));
    using f32x16 = float __attribute__((vector_size(16 * sizeof(float))));
    using bf16x16 = __bf16 __attribute__((vector_size(16 * sizeof(__bf16))));
    using bf16x32 = __bf16 __attribute__((vector_size(32 * sizeof(__bf16))));

    template <typename vector_t, typename scalar_t>
    [[gnu::always_inline]] vector_t load(const scalar_t* ptr) {
        vector_t result;
        std::memcpy(&result, ptr, sizeof(result));
        return result;
    }

    template <typename vector_t, typename scalar_t>
    [[gnu::always_inline]] void store(scalar_t* ptr, const vector_t& vector) {
        std::memcpy(ptr, &vector, sizeof(vector));
    }

    [[gnu::always_inline]] inline bf16x16 f32_to_bf16(const f32x16 value) {
        return _mm512_cvtneps_pbh(value);
    }

    [[gnu::always_inline]] inline f32x16 bf16_to_f32(const bf16x16 value) {
        return _mm512_cvtpbh_ps(value);
    }

} // namespace inference::cpu::avx
