#pragma once
#include "storage.hpp"
#include "tensor_shape.hpp"
#include "types/dtype.hpp"

#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace inference {
    class Tensor {
    public:
        [[nodiscard]] static Tensor empty(const TensorShape& shape, types::DType dtype, std::shared_ptr<allocator::Allocator> allocator) {
            auto storage = std::make_shared<Storage>(std::move(allocator), shape.element_count() * dtype_byte_size(dtype));
            return Tensor{ std::move(storage), shape, dtype };
        }

        template <typename T>
        [[nodiscard]]
        static Tensor from_vector(const std::vector<T>& values, const TensorShape& shape, types::DType dtype,
                                  std::shared_ptr<allocator::Allocator> allocator) {
            const auto expected_size_bytes = shape.element_count() * dtype_byte_size(dtype);
            const auto supplied_size_bytes = values.size() * sizeof(T);
            if (supplied_size_bytes != expected_size_bytes) {
                throw std::invalid_argument("tensor shape and dtype do not match the supplied data size");
            }

            auto result = empty(shape, dtype, std::move(allocator));
            auto destination = std::span<T>{ result.data<T>(), values.size() };
            std::ranges::uninitialized_copy(values, destination);
            return result;
        }

        [[nodiscard]] static Tensor from_storage(std::shared_ptr<Storage> storage, const TensorShape& shape, types::DType dtype) {
            if (!storage) {
                throw std::invalid_argument("tensor storage cannot be null");
            }

            Tensor result{ std::move(storage), shape, dtype };
            if (result.size_bytes() > result.storage_->size_bytes()) {
                throw std::invalid_argument("tensor shape exceeds the supplied storage");
            }
            return result;
        }

        [[nodiscard]] Tensor reshape(const TensorShape& shape) const {
            if (shape.element_count() != shape_.element_count()) {
                throw std::invalid_argument("reshape cannot change the number of tensor elements");
            }
            return Tensor{ storage_, shape, dtype_ };
        }

        [[nodiscard]] const TensorShape& shape() const {
            return shape_;
        }

        [[nodiscard]] std::size_t rank() const {
            return shape_.rank();
        }

        [[nodiscard]] std::size_t dim(const std::size_t dimension) const {
            if (dimension >= rank()) {
                throw std::out_of_range("tensor dimension is out of range");
            }
            return shape_[dimension];
        }

        [[nodiscard]] std::size_t num_elems() const {
            return shape_.element_count();
        }

        [[nodiscard]] std::size_t size_bytes() const {
            return num_elems() * dtype_byte_size(dtype_);
        }

        [[nodiscard]] types::DType dtype() const {
            return dtype_;
        }

        [[nodiscard]] types::Device device() const {
            return storage_->device();
        }

        template <typename T>
        [[nodiscard]] T* data() {
            return static_cast<T*>(storage_->data());
        }

        template <typename T>
        [[nodiscard]] const T* data() const {
            return static_cast<T*>(storage_->data());
        }

        [[nodiscard]] std::span<const std::byte> as_bytes() const {
            return { static_cast<const std::byte*>(storage_->data()), size_bytes() };
        }

        [[nodiscard]] std::span<std::byte> as_writable_bytes() {
            return { static_cast<std::byte*>(storage_->data()), size_bytes() };
        }

    private:
        Tensor(std::shared_ptr<Storage> storage, const TensorShape& shape, types::DType dtype)
            : storage_{ std::move(storage) },
              shape_{ shape },
              dtype_{ dtype } { }

        std::shared_ptr<Storage> storage_;
        TensorShape shape_;
        types::DType dtype_;
    };
} // namespace inference
