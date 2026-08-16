#pragma once
#include "backend/cpu/bf16.hpp"
#include "tensor/tensor.hpp"
#include <algorithm>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace test::util {
    struct RandomSeed {
        std::uint32_t value;
    };

    namespace cpu {
        namespace detail {
            template <typename Generator>
            [[nodiscard]] inference::Tensor random_bf16_tensor(const inference::TensorShape& shape, Generator& engine, const float min, const float max) {
                std::uniform_real_distribution distribution(min, max);
                std::vector<inference::cpu::bf16_t> values(shape.size());
                std::ranges::generate(values, [&] { return distribution(engine); });
                return inference::Tensor::from_host_bytes(std::as_bytes(std::span{ values }), shape, inference::types::DType::BF16);
            }
        } // namespace detail

        // todo maybe we should make this return both tensor and view at the same time? would make life a lil easier in tests.
        [[nodiscard]] inline inference::Tensor random_bf16_tensor(const inference::TensorShape& shape, const float min = -2.0F, const float max = 2.0F) {
            static std::mt19937 engine{ 0 };
            return detail::random_bf16_tensor(shape, engine, min, max);
        }

        [[nodiscard]] inline inference::Tensor random_bf16_tensor(const inference::TensorShape& shape, const RandomSeed seed, const float min = -2.0F,
                                                                  const float max = 2.0F) {
            std::mt19937 engine{ seed.value };
            return detail::random_bf16_tensor(shape, engine, min, max);
        }
    } // namespace cpu
} // namespace test::util
