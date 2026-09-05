#pragma once
#include "backend/cpu/bf16.hpp"
#include <concepts>
#include <cstddef>
#include <immintrin.h>
#include <type_traits>

namespace inference::cpu::avx {
    using f32x8 = float __attribute__((vector_size(8 * sizeof(float))));
    using f32x16 = float __attribute__((vector_size(16 * sizeof(float))));
    using bf16x16 = __bf16 __attribute__((vector_size(16 * sizeof(__bf16))));
    using bf16x32 = __bf16 __attribute__((vector_size(32 * sizeof(__bf16))));

    template <typename V> concept vector_t = std::same_as<V, f32x8> || std::same_as<V, f32x16> || std::same_as<V, bf16x16> || std::same_as<V, bf16x32>;
    template <typename S> concept scalar_t = std::same_as<S, float> || std::same_as<S, bf16_t>;

    template <vector_t V>
    inline constexpr std::size_t lanes_v = __builtin_vectorelements(V{});

    template <vector_t V, scalar_t S>
    [[nodiscard]] [[gnu::always_inline]] V broadcast(const S value) {
        using element_t = std::remove_cvref_t<decltype(V{}[0])>;
        return V{} + static_cast<element_t>(value);
    }

    template <vector_t V, scalar_t S>
    requires(sizeof(S) == sizeof(decltype(V{}[0])))
    [[nodiscard]] [[gnu::always_inline]] V load(const S* ptr) {
        V result;
        __builtin_memcpy_inline(&result, ptr, sizeof(result));
        return result;
    }

    template <vector_t V, scalar_t S>
    requires(sizeof(S) == sizeof(decltype(V{}[0])))
    [[gnu::always_inline]] void store(S* ptr, const V& vector) {
        __builtin_memcpy_inline(ptr, &vector, sizeof(vector));
    }

    [[nodiscard]] [[gnu::always_inline]] inline bf16x16 f32_to_bf16(const f32x16 value) {
        return _mm512_cvtneps_pbh(value);
    }

    [[nodiscard]] [[gnu::always_inline]] inline f32x16 bf16_to_f32(const bf16x16 value) {
        return _mm512_cvtpbh_ps(value);
    }

    [[nodiscard]] [[gnu::always_inline]] inline f32x16 dot_bf16(const bf16x32 a, const bf16x32 b, const f32x16 accumulator) noexcept {
        return _mm512_dpbf16_ps(accumulator, a, b);
    }

    [[nodiscard]] [[gnu::always_inline]] inline f32x16 load_bf16_as_f32(const bf16_t* ptr) {
        return bf16_to_f32(load<bf16x16>(ptr));
    }

    [[gnu::always_inline]] inline void store_f32_as_bf16(bf16_t* ptr, const f32x16 value) {
        store(ptr, f32_to_bf16(value));
    }

    [[nodiscard]] [[gnu::always_inline]] inline float reduce_add(const f32x16 vec) {
        return _mm512_reduce_add_ps(vec);
    }

    // generic fallback version of reduce_add since there is no builtin_reduce_add for floating types
    template <vector_t V, std::floating_point A = float>
    [[nodiscard]] [[gnu::always_inline]] A reduce_add(const V vec) {
        A result{};
        for (std::size_t index = 0; index < lanes_v<V>; index++) {
            result += static_cast<A>(vec[index]);
        }
        return result;
    }

    template <vector_t V>
    [[nodiscard]] [[gnu::always_inline]] auto reduce_max(const V vec) {
        return __builtin_reduce_max(vec);
    }

} // namespace inference::cpu::avx
