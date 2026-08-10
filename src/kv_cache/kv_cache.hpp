#pragma once
#include <memory>
#include <vector>

#include "tensor/tensor.hpp"

namespace inference {
    struct KVCache {
        // todo
        static constexpr std::size_t capacity = 1024;

        struct Layer {
            Tensor key;
            Tensor value;
        };

        std::vector<Layer> layers; // todo
        std::size_t token_count = 0;

        KVCache() = default;

        KVCache(const std::size_t layer_count, const std::size_t row_size, const types::DType dtype,
                const std::shared_ptr<allocator::BaseAllocator>& allocator) {
            layers.reserve(layer_count);
            for (std::size_t index = 0; index < layer_count; ++index) {
                layers.push_back({Tensor::empty({capacity, row_size}, dtype, allocator), Tensor::empty({capacity, row_size}, dtype, allocator)});
            }
        }

        void clear() noexcept {
            token_count = 0;
        }
    };
} // namespace inference
