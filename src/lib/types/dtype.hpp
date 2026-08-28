#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace inference::types {
    enum class DType : std::uint8_t {
        F16,
        BF16,
        F32,
        I32,
        I64,
    };

    [[nodiscard]] inline std::size_t dtype_byte_size(const DType dtype) {
        switch (dtype) {
        case DType::F16:
        case DType::BF16:
            return 2;
        case DType::I32:
        case DType::F32:
            return 4;
        case DType::I64:
            return 8;
        }

        throw std::invalid_argument("unknown dtype byte size");
    }

    [[nodiscard]] inline DType dtype_from_string(std::string_view str) {
        if (str == "I32") {
            return DType::I32;
        }

        if (str == "I64") {
            return DType::I64;
        }

        if (str == "BF16" || str == "bfloat16") {
            return DType::BF16;
        }

        if (str == "F16") {
            return DType::F16;
        }

        if (str == "F32") {
            return DType::F32;
        }

        throw std::invalid_argument("unknown dtype: " + std::string(str));
    }
} // namespace inference::types
