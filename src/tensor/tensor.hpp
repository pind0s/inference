#pragma once
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "storage.hpp"
#include "tensor_shape.hpp"
#include "types/dtype.hpp"
#include "util/move_only.hpp"

namespace inference {
    class Tensor : util::MoveOnly {
    public:
        [[nodiscard]] static Tensor empty(const TensorShape& shape, types::DType dtype, std::shared_ptr<allocator::BaseAllocator> allocator) {
            auto storage = std::make_shared<Storage>(std::move(allocator), shape.element_count() * dtype_size(dtype));
            return Tensor{std::move(storage), shape, dtype};
        }

        template <typename T>
        [[nodiscard]]
        static Tensor from_vector(const std::vector<T>& values, const TensorShape& shape, types::DType dtype, std::shared_ptr<allocator::BaseAllocator> allocator) {
            const auto expected_size_bytes = shape.element_count() * dtype_size(dtype);
            const auto supplied_size_bytes = values.size() * sizeof(T);
            if (supplied_size_bytes != expected_size_bytes) {
                throw std::invalid_argument("tensor shape and dtype do not match the supplied data size");
            }

            auto result = empty(shape, dtype, std::move(allocator));
            auto destination = std::span<T>{result.data<T>(), values.size()};
            std::ranges::uninitialized_copy(values, destination);
            return result;
        }

        [[nodiscard]] static Tensor from_storage(std::shared_ptr<Storage> storage, const TensorShape& shape, types::DType dtype) {
            if (!storage) {
                throw std::invalid_argument("tensor storage cannot be null");
            }

            Tensor result{std::move(storage), shape, dtype};
            if (result.size_bytes() > result.storage_->size_bytes()) {
                throw std::invalid_argument("tensor shape exceeds the supplied storage");
            }
            return result;
        }

        [[nodiscard]] Tensor reshape(const TensorShape& shape) const {
            if (shape.element_count() != shape_.element_count()) {
                throw std::invalid_argument("reshape cannot change the number of tensor elements");
            }
            return Tensor{storage_, shape, dtype_};
        }

        [[nodiscard]] const TensorShape& shape() const {
            return shape_;
        }

        [[nodiscard]] const TensorShape& strides() const {
            return strides_;
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
            return num_elems() * dtype_size(dtype_);
        }

        [[nodiscard]] types::DType dtype() const {
            return dtype_;
        }

        [[nodiscard]] types::Device device() const {
            return storage_->device();
        }

        [[nodiscard]] std::shared_ptr<Storage> storage() const {
            return storage_;
        }

        template <typename T>
        [[nodiscard]] T* data() {
            return static_cast<T*>(storage_->data());
        }

        template <typename T>
        [[nodiscard]] const T* data() const {
            return static_cast<T*>(storage_->data());
        }

        [[nodiscard]] void* raw_data() {
            return storage_->data();
        }

        [[nodiscard]] const void* raw_data() const {
            return storage_->data();
        }

        [[nodiscard]] std::span<const std::byte> as_bytes() const {
            return {static_cast<const std::byte*>(storage_->data()), size_bytes()};
        }

        [[nodiscard]] std::span<std::byte> as_writable_bytes() {
            return {static_cast<std::byte*>(storage_->data()), size_bytes()};
        }

    private:
        Tensor(std::shared_ptr<Storage> storage, const TensorShape& shape, types::DType dtype)
            : storage_{std::move(storage)}, shape_{shape}, strides_{make_contiguous_strides(shape_)}, dtype_{dtype} { }

        [[nodiscard]] static TensorShape make_contiguous_strides(const TensorShape& shape) {
            std::array<std::size_t, TensorShape::MAX_RANK> values{};
            std::size_t stride = 1;

            for (std::size_t index = shape.rank(); index > 0; --index) {
                values[index - 1] = stride;
                stride *= shape[index - 1];
            }
            return TensorShape{std::span<const std::size_t>{values.data(), shape.rank()}};
        }

        std::shared_ptr<Storage> storage_;
        TensorShape shape_;
        TensorShape strides_;
        types::DType dtype_;
    };
} // namespace inference
