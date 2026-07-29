#pragma once
#include <memory>

#include "allocator.hpp"

namespace inference {
    class CpuAllocator : public BaseAllocator {
    public:
        [[nodiscard]] void* allocate(std::size_t size_bytes) override {
            if (size_bytes == 0) {
                return nullptr;
            }
            return allocator_.allocate(size_bytes);
        }

        void deallocate(void* pointer, std::size_t size_bytes) noexcept override {
            allocator_.deallocate(static_cast<std::byte*>(pointer), size_bytes);
        }

        [[nodiscard]] Device device() const override {
            return Device::CPU;
        }

    private:
        std::allocator<std::byte> allocator_;
    };
} // namespace inference
