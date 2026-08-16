#pragma once
#include "tensor_shape.hpp"
#include "util/host_device.hpp"
#include <cassert>

namespace inference {
    template <typename T>
    class TensorView {
    public:
        constexpr TensorView(T* data, const TensorShape& shape) noexcept: data_(data), shape_(shape) { }

        [[nodiscard]] HOST_DEVICE constexpr T& operator[](const std::size_t offset) const noexcept {
            return data_[offset];
        }

        template <typename... Indices>
            requires(std::integral<Indices> && ...)
        [[nodiscard]] HOST_DEVICE constexpr T& operator()(const Indices... indices) const noexcept {
            assert(shape_.rank() == sizeof...(Indices));

            std::size_t dimension = 0;
            std::size_t offset = 0;
            ((offset += static_cast<std::size_t>(indices) * shape_.stride(dimension++)), ...);
            return data_[offset];
        }

        [[nodiscard]] HOST_DEVICE constexpr T* data() const noexcept {
            return data_;
        }

        [[nodiscard]] HOST_DEVICE constexpr std::size_t dim(const std::size_t dimension) const noexcept {
            return shape_.dim(dimension);
        }

        [[nodiscard]] HOST_DEVICE constexpr std::size_t size() const noexcept {
            return shape_.size();
        }

        [[nodiscard]] HOST_DEVICE constexpr auto begin() const noexcept {
            return data_;
        }

        [[nodiscard]] HOST_DEVICE constexpr auto end() const noexcept {
            return data_ + shape_.size();
        }

        [[nodiscard]] HOST_DEVICE constexpr const TensorShape& shape() const noexcept {
            return shape_;
        }

    private:
        T* data_ = nullptr;
        TensorShape shape_;
    };
} // namespace inference
