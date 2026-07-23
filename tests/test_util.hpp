#pragma once
#include <filesystem>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <util/files.hpp>

namespace test {
    [[nodiscard]]
    inline std::filesystem::path resource_path() {
        return std::filesystem::path{TEST_RESOURCE_DIR};
    }
} // namespace test
