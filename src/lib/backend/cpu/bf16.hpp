#pragma once
#include <cstdint>

namespace inference::cpu {
    class bf16_t {
    public:
        constexpr bf16_t() noexcept = default;

        /* implicit */ bf16_t(float value) noexcept; // NOLINT(*-explicit-conversions)

        /* implicit */ [[nodiscard]] operator float() const noexcept; // NOLINT(*-explicit-conversions)

        [[nodiscard]] float to_float() const noexcept;

        [[nodiscard]] static bf16_t from_float(float value) noexcept;

        [[nodiscard]] static constexpr bf16_t from_bits(const std::uint16_t bits) noexcept {
            bf16_t result;
            result.bits_ = bits;
            return result;
        }

        [[nodiscard]] constexpr std::uint16_t bits() const noexcept {
            return bits_;
        }

    private:
        std::uint16_t bits_{};
    };
} // namespace inference::cpu
