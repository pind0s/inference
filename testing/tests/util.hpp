#pragma once
#include <filesystem>

namespace test {
    [[nodiscard]] inline std::filesystem::path resource_path() {
        return std::filesystem::path{TEST_RESOURCE_DIR};
    }
} // namespace test
