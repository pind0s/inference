#pragma once

namespace inference::types {
    enum class DType : std::uint8_t {
        F16,
        BF16,
        F32,
    };

    [[nodiscard]] constexpr std::size_t element_size(const DType dtype) {
        switch (dtype) {
        case DType::F16:
        case DType::BF16:
            return 2;
        case DType::F32:
            return 4;
        }

        throw std::invalid_argument("Unknown size for safetensors dtype");
    }

    [[nodiscard]] inline DType dtype_from_string(std::string_view str) {
        if (str == "BF16") {
            return DType::BF16;
        }

        if (str == "F16") {
            return DType::F16;
        }

        if (str == "F32") {
            return DType::F32;
        }

        throw std::invalid_argument("Unsupported safetensors dtype: " + std::string(str));
    }
} // namespace inference::types
