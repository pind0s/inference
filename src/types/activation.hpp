#pragma once
#include <stdexcept>
#include <string>

namespace inference::types {
    enum class Activation {
        Silu,
        Relu,
    };

    [[nodiscard]] inline Activation activation_from_string(std::string_view str) {
        if (str == "silu") {
            return Activation::Silu;
        }

        if (str == "relu") {
            return Activation::Relu;
        }
        throw std::invalid_argument("Unsupported activation function: " + std::string(str));
    }
} // namespace inference::types
