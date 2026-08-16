#include "bf16.hpp"
#include <bit>
#include <immintrin.h>
#include <cstdint>

namespace inference::cpu {
    bf16_t::bf16_t(const float value) noexcept {
        const __bf16 native = _mm_cvtness_sbh(value);
        bits_ = std::bit_cast<std::uint16_t>(native);
    }

    bf16_t::operator float() const noexcept {
        return to_float();
    }

    float bf16_t::to_float() const noexcept {
        const auto native = std::bit_cast<__bf16>(bits_);
        return _mm_cvtsbh_ss(native);
    }

    bf16_t bf16_t::from_float(const float value) noexcept {
        return bf16_t{ value };
    }
} // namespace inference::cpu
