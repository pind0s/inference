#pragma once
#include "backend/cpu/bf16.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

namespace test {
    struct RandomSeed {
        std::uint32_t value;
    };

    template <typename T>
    struct RandomRange {
        T min;
        T max;
    };

    namespace detail {
        // bf16 has no std distribution, so sample in float and convert.
        template <typename T>
        using SampleType = std::conditional_t<std::is_same_v<T, inference::cpu::bf16_t>, float, T>;
    } // namespace detail

    // Uniformly distributed values in [range.min, range.max] (floating point: [min, max)). Deterministic for a given seed.
    template <typename T>
    [[nodiscard]] std::vector<T> random(const std::size_t count, const RandomSeed seed = { 0 }, const RandomRange<detail::SampleType<T>> range = { -2, 2 }) {
        using Sample = detail::SampleType<T>;
        using Distribution = std::conditional_t<std::is_floating_point_v<Sample>, std::uniform_real_distribution<Sample>, std::uniform_int_distribution<Sample>>;

        std::mt19937 engine{ seed.value };
        Distribution distribution(range.min, range.max);
        std::vector<T> values(count);
        std::ranges::generate(values, [&] { return static_cast<T>(distribution(engine)); });
        return values;
    }

    // A tensor of `shape` on `backend` filled by test::random<T>.
    template <typename T>
    [[nodiscard]] inference::Tensor random_tensor(const inference::TensorShape& shape, inference::Backend& backend, const RandomSeed seed = { 0 },
                                                  const RandomRange<detail::SampleType<T>> range = { -2, 2 }) {
        return inference::Tensor::from(random<T>(shape.size(), seed, range), shape, backend);
    }

    [[nodiscard]] inline inference::Tensor random_bf16(const inference::TensorShape& shape, inference::Backend& backend, const RandomSeed seed = { 0 },
                                                       const RandomRange<float> range = { -2, 2 }) {
        return random_tensor<inference::cpu::bf16_t>(shape, backend, seed, range);
    }
} // namespace test
