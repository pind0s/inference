#pragma once
#include <memory>
#include <stdexcept>
#include <utility>

#include "storage/memory_resource.hpp"

namespace inference {
    class Storage {
    public:
        Storage(std::shared_ptr<MemoryResource> resource, std::size_t size_bytes)
            : resource_{std::move(resource)}, size_bytes_{size_bytes} {
            if (!resource_) {
                throw std::invalid_argument("storage memory resource cannot be null");
            }
            data_ = resource_->allocate(size_bytes_);
        }

        ~Storage() {
            if (data_ != nullptr) {
                resource_->deallocate(data_, size_bytes_);
            }
        }

        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;

        [[nodiscard]] void* data() {
            return data_;
        }

        [[nodiscard]] const void* data() const {
            return data_;
        }

        [[nodiscard]] std::size_t size_bytes() const {
            return size_bytes_;
        }

        [[nodiscard]] Device device() const {
            return resource_->device();
        }

    private:
        std::shared_ptr<MemoryResource> resource_;
        void* data_ = nullptr;
        std::size_t size_bytes_;
    };
} // namespace inference
