#pragma once
#include "allocator.hpp"

namespace inference::allocator {
    class CpuAllocator : public BaseAllocator {
    public:
        static constexpr std::size_t alignment = 64;

        [[nodiscard]] void* allocate(std::size_t size_bytes) override {
            if (size_bytes == 0) {
                return nullptr;
            }

            return operator new(size_bytes, std::align_val_t{alignment});
        }

        void deallocate(void* pointer) noexcept override {
            operator delete(pointer, std::align_val_t{alignment});
        }

        [[nodiscard]] types::Device device() const override {
            return types::Device::CPU;
        }
    };
} // namespace inference::allocator
