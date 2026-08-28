#pragma once
#include "storage.hpp"
#include "tensor_shape.hpp"
#include "tensor_view.hpp"
#include "types/dtype.hpp"
#include <cstddef>
#include <memory>
#include <span>

namespace inference {
    class Tensor {
    public:
        [[nodiscard]] static Tensor empty(const TensorShape& shape, types::DType dtype, Backend& backend);

        [[nodiscard]] TensorShape shape() const {
            return shape_;
        }

        [[nodiscard]] std::size_t size() const noexcept {
            return shape_.size();
        }

        [[nodiscard]] std::size_t size_bytes() const noexcept {
            return storage_->size_bytes();
        }

        [[nodiscard]] types::Device device() const noexcept {
            return storage_->backend.device();
        }

        [[nodiscard]] types::DType dtype() const noexcept {
            return dtype_;
        }

        template <typename T>
        [[nodiscard]] T* data() noexcept {
            return static_cast<T*>(storage_->ptr);
        }

        template <typename T>
        [[nodiscard]] const T* data() const noexcept {
            return static_cast<const T*>(storage_->ptr);
        }

        template <typename T>
        [[nodiscard]] TensorView<T> view() noexcept {
            return TensorView<T>{ data<T>(), shape_ };
        }

        template <typename T>
        [[nodiscard]] TensorView<const T> view() const noexcept {
            return TensorView<const T>{ data<T>(), shape_ };
        }

        [[nodiscard]] bool is_cuda() const {
            return device() == types::Device::CUDA;
        }

        [[nodiscard]] bool is_cpu() const {
            return device() == types::Device::CPU;
        }

        [[nodiscard]] std::span<std::byte> bytes() {
            return { static_cast<std::byte*>(storage_->ptr), storage_->size_bytes() };
        }

        [[nodiscard]] std::span<const std::byte> bytes() const {
            return { static_cast<const std::byte*>(storage_->ptr), storage_->size_bytes() };
        }

    private:
        Tensor(const TensorShape& shape, types::DType dtype, Backend& backend);

        std::shared_ptr<Storage> storage_;
        TensorShape shape_;
        types::DType dtype_;
    };

} // namespace inference
