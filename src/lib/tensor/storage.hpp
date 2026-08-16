#pragma once
#include "backend/backend.hpp"
#include <new>
#include <utility>

namespace inference {
    class Storage {
    public:
        Backend& backend;
        void* ptr;
        std::size_t bytes;

        Storage(Backend& backend, std::size_t bytes): backend(backend), ptr(bytes == 0 ? nullptr : backend.allocate(bytes)), bytes(bytes) {
            if (bytes != 0 && ptr == nullptr) {
                throw std::bad_alloc{};
            }
        }

        ~Storage() noexcept {
            if (ptr != nullptr) {
                backend.deallocate(ptr, bytes);
            }
        }

        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;
        Storage(Storage&& other) noexcept: backend(other.backend), ptr(std::exchange(other.ptr, nullptr)), bytes(std::exchange(other.bytes, 0)) { }

        Storage& operator=(Storage&&) = delete;

        [[nodiscard]] std::size_t size_bytes() const noexcept {
            return bytes;
        }
    };
} // namespace inference
