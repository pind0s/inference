#pragma once
#include <bit>
#include <cstdint>

namespace inference::cpu {
    class bf16_t {
    public:
        constexpr bf16_t() noexcept = default;

        /* implicit */ bf16_t(const float value) noexcept // NOLINT(*-explicit-conversions)
            : bits_{ float_to_bits(value) } { }

        /* implicit */ [[nodiscard]] operator float() const noexcept { // NOLINT(*-explicit-conversions)
            return to_float();
        }

        [[nodiscard]] float to_float() const noexcept {
            return std::bit_cast<float>(static_cast<std::uint32_t>(bits_) << 16U);
        }

        [[nodiscard]] static bf16_t from_float(const float value) noexcept {
            return bf16_t{ value };
        }

    private:
        // We can't use _mm_cvtness_sbh/_mm_cvtsbh_ss since we will include this file in tests so we can compare cuda kernels to references :/
        [[nodiscard]] static std::uint16_t float_to_bits(const float value) noexcept {
            auto bits = std::bit_cast<std::uint32_t>(value);

            constexpr std::uint32_t exponent_mask = 0x7F80'0000;
            constexpr std::uint32_t magnitude_mask = 0x7FFF'FFFF;
            constexpr std::uint16_t quiet_nan_bit = 0x0040;
            if ((bits & magnitude_mask) > exponent_mask) {
                return static_cast<std::uint16_t>((bits >> 16) | quiet_nan_bit);
            }

            constexpr std::uint32_t rounding_bias = 0x7FFF;
            const auto retained_lsb = (bits >> 16) & 1;
            bits += rounding_bias + retained_lsb;
            return static_cast<std::uint16_t>(bits >> 16);
        }

        std::uint16_t bits_{};
    };
} // namespace inference::cpu
