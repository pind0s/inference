#pragma once
#include "backend/cpu/bf16.hpp"
#include <cstring>
#include <immintrin.h>
#include <type_traits>

namespace inference::cpu::avx {
    using f32x8 = float __attribute__((vector_size(8 * sizeof(float))));
    using f32x16 = float __attribute__((vector_size(16 * sizeof(float))));
    using bf16x16 = __bf16 __attribute__((vector_size(16 * sizeof(__bf16))));
    using bf16x32 = __bf16 __attribute__((vector_size(32 * sizeof(__bf16))));

    template <typename vector_t, typename scalar_t>
    [[nodiscard]] [[gnu::always_inline]] vector_t broadcast(const scalar_t value) noexcept {
        using element_t = std::remove_cvref_t<decltype(vector_t{}[0])>;
        return vector_t{} + static_cast<element_t>(value);
    }

    template <typename vector_t, typename scalar_t>
    [[nodiscard]] [[gnu::always_inline]] vector_t load(const scalar_t* ptr) noexcept {
        vector_t result;
        std::memcpy(&result, ptr, sizeof(result));
        return result;
    }

    template <typename vector_t, typename scalar_t>
    [[gnu::always_inline]] void store(scalar_t* ptr, const vector_t& vector) noexcept {
        std::memcpy(ptr, &vector, sizeof(vector));
    }

    [[nodiscard]] [[gnu::always_inline]] inline bf16x16 f32_to_bf16(const f32x16 value) noexcept {
        return _mm512_cvtneps_pbh(value);
    }

    [[nodiscard]] [[gnu::always_inline]] inline f32x16 bf16_to_f32(const bf16x16 value) noexcept {
        return _mm512_cvtpbh_ps(value);
    }

    [[nodiscard]] [[gnu::always_inline]] inline f32x16 zero() noexcept {
        return _mm512_setzero_ps();
    }

    [[nodiscard]] [[gnu::always_inline]] inline f32x16 dot_bf16(const bf16x32 a, const bf16x32 b, const f32x16 accumulator) noexcept {
        return _mm512_dpbf16_ps(accumulator, a, b);
    }

    [[nodiscard]] [[gnu::always_inline]] inline f32x16 load_bf16_as_f32(const bf16_t* ptr) noexcept {
        return bf16_to_f32(load<bf16x16>(ptr));
    }

    [[gnu::always_inline]] inline void store_f32_as_bf16(bf16_t* ptr, const f32x16 value) noexcept {
        store(ptr, f32_to_bf16(value));
    }

    [[nodiscard]] [[gnu::always_inline]] inline float reduce_add(const f32x16 value) noexcept {
        return _mm512_reduce_add_ps(value);
    }

    [[nodiscard]] [[gnu::always_inline]] inline float reduce_max(const f32x16 value) noexcept {
        return _mm512_reduce_max_ps(value);
    }

} // namespace inference::cpu::avx
