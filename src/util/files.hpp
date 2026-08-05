#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <vector>

// slightly modified https://github.com/es3n1n/common/blob/master/include/es3n1n/common/files.hpp
namespace util {
    [[nodiscard]] inline std::filesystem::path require_file(const std::filesystem::path& path) {
        if (!std::filesystem::is_regular_file(path)) {
            throw std::runtime_error("Required file does not exist: " + path.string());
        }

        return path;
    }

    inline std::optional<std::size_t> file_size(std::ifstream& file) {
        if (!file.good()) {
            return {};
        }

        file.seekg(0, std::ios::end);
        const auto file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        return file_size;
    }

    inline std::optional<std::vector<std::byte>> read_file(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.good()) {
            return {};
        }

        std::vector<std::byte> buffer(file_size(file).value());
        file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        return buffer;
    }

    inline void write_file(const std::filesystem::path& path, const std::uint8_t* raw_buffer, const std::size_t buffer_size) {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(raw_buffer), static_cast<std::streamsize>(buffer_size));
    }
} // namespace util
