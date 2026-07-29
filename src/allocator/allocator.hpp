#pragma once
#include "types/device.hpp"

namespace inference {
    class BaseAllocator {
    public:
        virtual ~BaseAllocator() = default;

        [[nodiscard]] virtual void* allocate(std::size_t size_bytes) = 0;
        virtual void deallocate(void* pointer, std::size_t size_bytes) noexcept = 0;

        [[nodiscard]] virtual Device device() const = 0;
    };
} // namespace inference
