#pragma once
#include "tensor/tensor_view.hpp"
#include <algorithm>
#include <cstddef>

namespace inference::cpu::kernels {
    template <typename T>
    void kv_cache_update(TensorView<const T> key, TensorView<const T> value, TensorView<T> cached_key, TensorView<T> cached_value,
                         std::size_t token_offset) {
        std::ranges::copy(key, cached_key.begin() + token_offset * key.size());
        std::ranges::copy(value, cached_value.begin() + token_offset * value.size());
    }
} // namespace inference::cpu::kernels
