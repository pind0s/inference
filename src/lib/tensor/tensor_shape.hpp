#pragma once
#include "util/host_device.hpp"
#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>

namespace inference {
    class TensorShape {
    public:
        static constexpr std::size_t max_rank = 4;

        constexpr TensorShape(std::initializer_list<std::size_t> dimensions): TensorShape{ std::span(dimensions) } { }

        // NOLINTNEXTLINE(*-explicit-conversions)
        /* implicit */ constexpr TensorShape(const std::span<const std::size_t> dimensions): rank_{ static_cast<std::uint16_t>(dimensions.size()) } {
            if (dimensions.size() > max_rank) {
                throw std::length_error("tensor rank cannot exceed " + std::to_string(max_rank) + " dims");
            }

            num_elems_ = 1;
            for (int i = 0; i < dimensions.size(); i++) {
                shape_[i] = dimensions[i];
                num_elems_ *= shape_[i];
            }

            stride_[rank_ - 1] = 1;
            for (std::size_t i = rank_ - 1; i != 0; --i) {
                stride_[i - 1] = stride_[i] * shape_[i];
            }
        }

        [[nodiscard]] HOST_DEVICE constexpr std::size_t rank() const {
            return rank_;
        }

        [[nodiscard]] HOST_DEVICE constexpr std::size_t dim(const std::size_t dimension) const {
            return shape_[dimension];
        }

        [[nodiscard]] HOST_DEVICE constexpr std::size_t stride(const std::size_t dimension) const {
            return stride_[dimension];
        }

        [[nodiscard]] HOST_DEVICE constexpr std::size_t size() const {
            return num_elems_;
        }

        [[nodiscard]] std::string to_string() const {
            std::string result;
            result += "shape: [";

            for (std::size_t i = 0; i < rank_; i++) {
                result += std::to_string(shape_[i]);
                if (i < rank_ - 1) {
                    result += ", ";
                }
            }
            result += "]\n";
            result += "num elems: " + std::to_string(num_elems_);
            return result;
        }

    private:
        std::uint32_t rank_ = 0;
        // can't use std::array because we access this in device code
        std::uint32_t shape_[max_rank] = {};
        std::uint32_t stride_[max_rank] = {};
        std::size_t num_elems_ = 0;
    };
} // namespace inference