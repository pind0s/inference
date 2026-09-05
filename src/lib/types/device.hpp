#pragma once
#include <cstdint>
#include <string>

namespace inference::types {
    enum class Device : std::uint8_t {
        CPU,
        CUDA,
    };

    inline std::string device_to_string(const Device& device) {
        switch (device) {
        case Device::CPU:
            return "CPU";
        case Device::CUDA:
            return "CUDA";
        }
    }
} // namespace inference::types
