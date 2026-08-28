#pragma once
#include "tensor/tensor.hpp"
#include <vector>

namespace inference {
    struct KVCache {
        struct Layer {
            Tensor key;
            Tensor value;
        };

        std::vector<Layer> layers;
        std::size_t capacity = 0;
        std::size_t token_count = 0;

        KVCache(std::size_t layer_count, std::size_t num_heads, std::size_t head_dim, std::size_t capacity, types::DType dtype, Backend& backend)
            : capacity(capacity) {
            layers.reserve(layer_count);
            for (std::size_t index = 0; index < layer_count; ++index) {
                auto key = Tensor::empty({ capacity, num_heads, head_dim }, dtype, backend);
                auto value = Tensor::empty({ capacity, num_heads, head_dim }, dtype, backend);
                layers.emplace_back(std::move(key), std::move(value));
            }
        }

        void reset() noexcept {
            token_count = 0;
        }
    };
} // namespace inference
