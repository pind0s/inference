#pragma once
#include "types/device.hpp"

namespace inference::allocator {
    class Allocator {
    public:
        virtual ~Allocator() = default;

        [[nodiscard]] virtual void* allocate(std::size_t size_bytes) = 0;
        virtual void deallocate(void* pointer) noexcept = 0;

        [[nodiscard]] virtual types::Device device() const = 0;
    };
} // namespace inference::allocator
