#pragma once
#include "backend/backend.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <vector>

namespace inference::backend {
    struct RopeCacheView {
        TensorView<const float> cosine;
        TensorView<const float> sine;
    };

    struct RopeCache {
        Tensor cosine;
        Tensor sine;

        [[nodiscard]] RopeCacheView view() const noexcept {
            return { .cosine = cosine.view<float>(), .sine = sine.view<float>() };
        }

        [[nodiscard]] RopeCache to_host() const {
            return { .cosine = cosine.to_host(), .sine = sine.to_host() };
        }
    };

    [[nodiscard]] inline RopeCache make_rope_cache(std::size_t max_position, std::size_t head_size, float theta, Backend& backend) {
        const auto half_size = head_size / 2;
        std::vector<float> cos(max_position * half_size);
        std::vector<float> sin(max_position * half_size);
        std::vector<float> inverse_frequencies(half_size);

        for (std::size_t index = 0; index < half_size; ++index) {
            const auto exponent = 2.0F * static_cast<float>(index) / static_cast<float>(head_size);
            inverse_frequencies[index] = 1.0F / std::pow(theta, exponent);
        }

        for (std::size_t position = 0; position < max_position; ++position) {
            for (std::size_t index = 0; index < half_size; ++index) {
                const auto angle = static_cast<float>(position) * inverse_frequencies[index];
                cos[position * half_size + index] = std::cos(angle);
                sin[position * half_size + index] = std::sin(angle);
            }
        }

        return { .cosine = Tensor::from(cos, { max_position, half_size }, backend), .sine = Tensor::from(sin, { max_position, half_size }, backend) };
    }
} // namespace inference::backend
