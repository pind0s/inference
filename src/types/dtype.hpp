#pragma once

namespace inference {
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
        default:
            std::unreachable();
        }
    }
} // namespace inference
