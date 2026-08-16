#pragma once
#include <cstdint>

namespace inference::types {
    enum class Device : std::uint8_t {
        CPU,
        CUDA,
    };
} // namespace inference::types
