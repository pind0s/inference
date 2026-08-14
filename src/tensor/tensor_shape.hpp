#pragma once
#include <array>
#include <span>

namespace inference {
    class TensorShape {
    public:
        static constexpr std::size_t max_rank = 5;

        constexpr TensorShape() noexcept = default;

        constexpr TensorShape(std::initializer_list<std::size_t> dimensions): TensorShape{ std::span(dimensions) } { }

        constexpr TensorShape(const std::span<const std::size_t> dimensions): rank_{ dimensions.size() } {
            if (dimensions.size() > max_rank) {
                throw std::length_error("tensor rank cannot exceed 5 dimensions");
            }
            std::ranges::copy(dimensions, dimensions_.begin());
        }

        [[nodiscard]] constexpr std::size_t rank() const {
            return rank_;
        }

        [[nodiscard]] constexpr std::size_t operator[](const std::size_t index) const {
            return dimensions_[index];
        }

        [[nodiscard]] constexpr std::size_t element_count() const noexcept {
            std::size_t result = 1;
            for (std::size_t i = 0; i < rank_; ++i) {
                result *= dimensions_[i];
            }
            return result;
        }

        constexpr bool operator==(const TensorShape&) const = default;

    private:
        std::array<std::size_t, max_rank> dimensions_ = {};
        std::size_t rank_ = 0;
    };
} // namespace inference
