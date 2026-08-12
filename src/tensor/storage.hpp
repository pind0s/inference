#pragma once
#include "allocator/allocator.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace inference {
    class Storage {
    public:
        Storage(std::shared_ptr<allocator::Allocator> resource, std::size_t size_bytes): allocator_{ std::move(resource) }, size_bytes_{ size_bytes } {
            if (!allocator_) {
                throw std::invalid_argument("allocator for storage is nullptr");
            }
            data_ = allocator_->allocate(size_bytes_);
        }

        ~Storage() {
            if (data_ != nullptr) {
                allocator_->deallocate(data_);
            }
        }

        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;
        Storage(Storage&&) = delete;
        Storage& operator=(Storage&&) = delete;

        [[nodiscard]] void* data() {
            return data_;
        }

        [[nodiscard]] const void* data() const {
            return data_;
        }

        [[nodiscard]] std::size_t size_bytes() const {
            return size_bytes_;
        }

        [[nodiscard]] types::Device device() const {
            return allocator_->device();
        }

    private:
        std::shared_ptr<allocator::Allocator> allocator_;
        void* data_ = nullptr;
        std::size_t size_bytes_;
    };
} // namespace inference
