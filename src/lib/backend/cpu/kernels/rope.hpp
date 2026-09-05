#pragma once
#include "backend/cpu/bf16.hpp"
#include "backend/util/rope_cache.hpp"
#include "tensor/tensor_view.hpp"

namespace inference::cpu::kernels {
    inline void rope(const TensorView<bf16_t>& values, const backend::RopeCacheView cache, const std::size_t position) {
        const auto head_count = values.dim(0);
        const auto head_size = values.dim(1);
        const auto half_size = head_size / 2;

        for (std::size_t head = 0; head < head_count; ++head) {
            for (std::size_t index = 0; index < half_size; ++index) {
                const auto cosine = cache.cosine(position, index);
                const auto sine = cache.sine(position, index);
                const auto x = values(head, index).to_float();
                const auto y = values(head, half_size + index).to_float();
                values(head, index) = x * cosine - y * sine;
                values(head, half_size + index) = y * cosine + x * sine;
            }
        }
    }

} // namespace inference::cpu::kernels
