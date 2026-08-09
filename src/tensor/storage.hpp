#pragma once
#include <memory>
#include <stdexcept>
#include <utility>

#include "allocator/allocator.hpp"
#include "util/move_only.hpp"

namespace inference {
    // todo I don't think Storage is move only? i think we need to redesign this
    class Storage : util::MoveOnly {
    public:
        Storage(std::shared_ptr<allocator::BaseAllocator> resource, std::size_t size_bytes)
            : resource_{std::move(resource)}, size_bytes_{size_bytes} {
            if (!resource_) {
                throw std::invalid_argument("storage memory resource cannot be null");
            }
            data_ = resource_->allocate(size_bytes_);
        }

        ~Storage() {
            if (data_ != nullptr) {
                resource_->deallocate(data_);
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

        [[nodiscard]] types::Device device() const {
            return resource_->device();
        }

    private:
        std::shared_ptr<allocator::BaseAllocator> resource_;
        void* data_ = nullptr;
        std::size_t size_bytes_;
    };
} // namespace inference
