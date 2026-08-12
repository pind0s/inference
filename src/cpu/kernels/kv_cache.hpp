#pragma once
#include <algorithm>
#include <cstddef>
#include <span>

namespace inference::cpu::kernels {
    inline void kv_cache_update(const std::span<const std::byte> key, const std::span<const std::byte> value, const std::span<std::byte> cached_key,
                                const std::span<std::byte> cached_value, const std::size_t offset_bytes) {
        std::ranges::copy(key, cached_key.begin() + offset_bytes);
        std::ranges::copy(value, cached_value.begin() + offset_bytes);
    }
} // namespace inference::cpu::kernels
