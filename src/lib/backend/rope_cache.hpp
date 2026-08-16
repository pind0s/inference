#pragma once
#include "tensor/tensor.hpp"
#include <cmath>
#include <utility>
#include <vector>

// todo put this under some namespace
namespace inference {
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
    };

    [[nodiscard]] inline RopeCache make_rope_cache(std::size_t max_position, std::size_t head_size, float theta, Backend& backend) {
        auto& cpu_backend = backend.device() == types::Device::CPU ? backend : Backend::get_backend<types::Device::CPU>();
        const auto half_size = head_size / 2;
        const TensorShape shape{ max_position, half_size };

        auto cos_tensor = Tensor::empty(shape, types::DType::F32, cpu_backend);
        auto sin_tensor = Tensor::empty(shape, types::DType::F32, cpu_backend);

        auto cos_view = cos_tensor.view<float>();
        auto sin_view = sin_tensor.view<float>();

        auto inverse_frequencies = std::vector<float>(half_size);
        for (std::size_t index = 0; index < half_size; ++index) {
            const auto exponent = 2.0F * static_cast<float>(index) / static_cast<float>(head_size);
            inverse_frequencies[index] = 1.0F / std::pow(theta, exponent);
        }

        for (std::size_t position = 0; position < max_position; ++position) {
            for (std::size_t index = 0; index < half_size; ++index) {
                const auto angle = static_cast<float>(position) * inverse_frequencies[index];
                cos_view(position, index) = std::cos(angle);
                sin_view(position, index) = std::sin(angle);
            }
        }

        if (backend.device() == types::Device::CUDA) {
            cos_tensor = cos_tensor.host_to_device();
            sin_tensor = sin_tensor.host_to_device();
        }

        return { .cosine = std::move(cos_tensor), .sine = std::move(sin_tensor) };
    }
} // namespace inference
