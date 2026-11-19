#pragma once
#include "storage.hpp"
#include "tensor_shape.hpp"
#include "tensor_view.hpp"
#include "types/dtype.hpp"
#include <array>
#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>

namespace inference {
    class Tensor {
    public:
        [[nodiscard]] static Tensor empty(const TensorShape& shape, const types::DType dtype, Backend& backend) {
            return { shape, dtype, backend };
        }

        // Allocates a tensor on `backend` and copies `values` into it. The dtype is deduced from the element type.
        template <typename Range>
            requires std::ranges::contiguous_range<const Range> && std::ranges::sized_range<const Range>
        [[nodiscard]] static Tensor from(const Range& values, const TensorShape& shape, Backend& backend) {
            using Value = std::remove_cv_t<std::ranges::range_value_t<const Range>>;
            return backend.make_tensor(std::as_bytes(std::span{ values }), shape, types::dtype_of_v<Value>);
        }

        // Returns a CPU copy of this tensor. Waits for all work queued on the tensor's backend before copying.
        // Always deep-copies, so the result can be modified without touching this tensor.
        [[nodiscard]] Tensor to_host() const {
            auto result = empty(shape_, dtype_, Backend::get_backend<types::Device::CPU>());
            storage_->backend.copy_to_host(*this, result.bytes());
            return result;
        }

        // Select one entry along the first dimension without copying data.
        [[nodiscard]] Tensor select(const std::size_t index) {
            if (shape_.rank() < 2) {
                throw std::invalid_argument("Tensor::select requires at least two dimensions");
            }
            if (index >= shape_.dim(0)) {
                throw std::out_of_range("Tensor::select index is out of bounds");
            }

            std::array<std::size_t, TensorShape::max_rank> dimensions{};
            for (std::size_t dim = 1; dim < shape_.rank(); ++dim) {
                dimensions[dim - 1] = shape_.dim(dim);
            }

            auto result = *this;
            result.byte_offset_ += index * shape_.stride(0) * types::dtype_byte_size(dtype_);
            result.shape_ = TensorShape{ std::span{ dimensions }.first(shape_.rank() - 1) };
            return result;
        }

        [[nodiscard]] TensorShape shape() const {
            return shape_;
        }

        [[nodiscard]] std::size_t size() const noexcept {
            return shape_.size();
        }

        [[nodiscard]] std::size_t size_bytes() const noexcept {
            return shape_.size() * types::dtype_byte_size(dtype_);
        }

        [[nodiscard]] types::Device device() const noexcept {
            return storage_->backend.device();
        }

        [[nodiscard]] types::DType dtype() const noexcept {
            return dtype_;
        }

        template <typename T>
        [[nodiscard]] T* data() noexcept {
            return static_cast<T*>(storage_->data(byte_offset_));
        }

        template <typename T>
        [[nodiscard]] const T* data() const noexcept {
            return static_cast<const T*>(storage_->data(byte_offset_));
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
            return { data<std::byte>(), size_bytes() };
        }

        [[nodiscard]] std::span<const std::byte> bytes() const {
            return { data<std::byte>(), size_bytes() };
        }

    private:
        Tensor(const TensorShape& shape, const types::DType dtype, Backend& backend)
            : storage_(std::make_shared<Storage>(backend, shape.size() * types::dtype_byte_size(dtype))),
              shape_(shape),
              dtype_(dtype) { }

        std::shared_ptr<Storage> storage_;
        TensorShape shape_;
        types::DType dtype_;
        std::size_t byte_offset_ = 0;
    };

} // namespace inference
