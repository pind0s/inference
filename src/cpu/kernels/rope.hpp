#pragma once
#include <cmath>

namespace inference::cpu::kernels {
    inline void rope(__bf16* __restrict values, const std::size_t head_count, const std::size_t head_size, const float theta, const std::size_t pos) {
        const auto half_size = head_size / 2;

        for (std::size_t head = 0; head < head_count; ++head) {
            const auto base = head * head_size;
            for (std::size_t index = 0; index < half_size; ++index) {
                const auto exponent = static_cast<float>(2 * index) / static_cast<float>(head_size);
                const auto angle = static_cast<float>(pos) / std::pow(theta, exponent);
                const auto cosine = std::cos(angle);
                const auto sine = std::sin(angle);
                const auto first = static_cast<float>(values[base + index]);
                const auto second = static_cast<float>(values[base + half_size + index]);

                values[base + index] = static_cast<__bf16>(first * cosine - second * sine);
                values[base + half_size + index] = static_cast<__bf16>(second * cosine + first * sine);
            }
        }
    }
} // namespace inference::cpu::kernels
