#pragma once
#include "backend/cpu/bf16.hpp"
#include "backend/util/rope_cache.hpp"
#include "util/to_host.hpp"

namespace test::reference {
    [[nodiscard]] inline inference::Tensor rope(const inference::Tensor& input, const inference::backend::RopeCache& cache, const std::size_t position) {
        using namespace inference;
        auto result = to_host(input);
        const auto host_cache = to_host(cache);
        const auto head_count = input.shape().dim(0);
        const auto head_size = input.shape().dim(1);
        const auto half_size = head_size / 2;
        auto output = result.view<cpu::bf16_t>();
        const auto cache_values = host_cache.view();

        for (std::size_t head = 0; head < head_count; ++head) {
            for (std::size_t index = 0; index < half_size; ++index) {
                const auto cosine = cache_values.cosine(position, index);
                const auto sine = cache_values.sine(position, index);
                const auto x = output(head, index).to_float();
                const auto y = output(head, half_size + index).to_float();
                output(head, index) = x * cosine - y * sine;
                output(head, half_size + index) = y * cosine + x * sine;
            }
        }
        return result;
    }
} // namespace test::reference
