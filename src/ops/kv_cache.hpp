#pragma once
#include <stdexcept>

#include "cpu/kernels/kv_cache.hpp"
#include "tensor/tensor.hpp"

namespace inference::ops {
    inline void kv_cache_update(const Tensor& key, const Tensor& value, Tensor& cached_key, Tensor& cached_value,
                                const std::size_t token_offset) {
        const auto offset_bytes = token_offset * key.size_bytes();

        switch (key.device()) {
        case types::Device::CPU:
            cpu::kernels::kv_cache_update(key.as_bytes(), value.as_bytes(), cached_key.as_writable_bytes(), cached_value.as_writable_bytes(),
                                          offset_bytes);
            return;
        case types::Device::CUDA:
            throw std::runtime_error("cuda not implemented");
        }
    }
} // namespace inference::ops
