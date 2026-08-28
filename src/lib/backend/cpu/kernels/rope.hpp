#pragma once
#include "backend/cpu/bf16.hpp"
#include "backend/util/rope_cache.hpp"
#include "tensor/tensor_view.hpp"

namespace inference::cpu::kernels {
    // todo fuse rope so we do q, k at the same time
    inline void rope(const TensorView<bf16_t>& values, const RopeCacheView cache, const std::size_t position) {
        const auto head_count = values.dim(0);
        const auto head_size = values.dim(1);
        const auto half_size = head_size / 2;

        for (std::size_t head = 0; head < head_count; ++head) {
            for (std::size_t index = 0; index < half_size; ++index) {
                const auto cos = cache.cosine(position, index);
                const auto sin = cache.sine(position, index);
                const auto first = values(head, index).to_float();
                const auto second = values(head, half_size + index).to_float();
                values(head, index) = first * cos - second * sin;
                values(head, half_size + index) = second * cos + first * sin;
            }
        }
    }
} // namespace inference::cpu::kernels
