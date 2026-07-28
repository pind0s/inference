#pragma once
#include <memory>

#include "storage/memory_resource.hpp"

namespace inference {
    class CpuMemoryResource final : public MemoryResource {
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
