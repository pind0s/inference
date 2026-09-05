#pragma once
#include "backend/cpu/bf16.hpp"
#include "tensor/tensor.hpp"
#include <cstdint>
#include <ranges>
#include <span>
#include <type_traits>

namespace test {
    namespace detail {
        // todo move this to library
        template <typename T>
        constexpr inference::types::DType dtype_for() {
            using inference::types::DType;
            if constexpr (std::is_same_v<T, inference::cpu::bf16_t>) {
                return DType::BF16;
            } else if constexpr (std::is_same_v<T, float>) {
                return DType::F32;
            } else if constexpr (std::is_same_v<T, std::int32_t>) {
                return DType::I32;
            } else if constexpr (std::is_same_v<T, std::int64_t>) {
                return DType::I64;
            } else {
                static_assert(sizeof(T) == 0, "unsupported tensor element type");
            }
        }
    } // namespace detail

    template <typename Range>
        requires std::ranges::contiguous_range<const Range> && std::ranges::sized_range<const Range>
    [[nodiscard]] inference::Tensor make_tensor(const Range& values, const inference::TensorShape& shape, inference::Backend& backend) {
        using Value = std::ranges::range_value_t<const Range>;
        return backend.make_tensor(std::as_bytes(std::span{ values }), shape, detail::dtype_for<Value>());
    }
} // namespace test
