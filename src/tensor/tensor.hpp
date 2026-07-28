#pragma once

#include "tensor_shape.hpp"
#include "types/dtype.hpp"

namespace inference {
    class Tensor {
    public:
        [[nodiscard]] static Tensor from_storage(std::shared_ptr<Storage> storage, const TensorShape& shape, DType dtype) {
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
            if (element_count(shape) != num_elems()) {
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

        [[nodiscard]] std::size_t num_elems() const {
            return element_count(shape_);
        }

        [[nodiscard]] std::size_t size_bytes() const {
            return num_elems() * element_size(dtype_);
        }

        [[nodiscard]] DType dtype() const {
            return dtype_;
        }

        [[nodiscard]] Device device() const {
            return storage_->device();
        }

        [[nodiscard]] void* raw_data() {
            return storage_->data();
        }

        [[nodiscard]] const void* raw_data() const {
            return storage_->data();
        }

        template <typename T>
        [[nodiscard]] std::span<T> as_span() {
            // todo verify size of T and stored dtype
            return std::span<T>(static_cast<T*>(storage_->data()), num_elems());
        }

    private:
        Tensor(std::shared_ptr<Storage> storage, const TensorShape& shape, DType dtype)
            : storage_{std::move(storage)}, shape_{shape}, strides_{make_contiguous_strides(shape_)}, dtype_{dtype} { }

        [[nodiscard]] static std::size_t element_count(const TensorShape& shape) noexcept {
            std::size_t result = 1;
            for (const auto dimension : shape) {
                result *= dimension;
            }
            return result;
        }

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
        DType dtype_;
    };
} // namespace inference
