#pragma once
#include "backend/backend.hpp"
#include <gtest/gtest.h>
#include <string>
#include <utility>

namespace test {
    class BackendTest : public testing::TestWithParam<inference::types::Device> {
    protected:
        [[nodiscard]] static inference::Backend& backend() {
            switch (GetParam()) {
            case inference::types::Device::CPU:
                return inference::Backend::get_backend<inference::types::Device::CPU>();
            case inference::types::Device::CUDA:
                return inference::Backend::get_backend<inference::types::Device::CUDA>();
            }
            std::unreachable();
        }
    };

    [[nodiscard]] inline std::string backend_name(const testing::TestParamInfo<inference::types::Device>& info) {
        switch (info.param) {
        case inference::types::Device::CPU:
            return "CPU";
        case inference::types::Device::CUDA:
            return "CUDA";
        }
        std::unreachable();
    }
} // namespace test
