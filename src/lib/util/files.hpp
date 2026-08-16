#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

// slightly modified https://github.com/es3n1n/common/blob/master/include/es3n1n/common/files.hpp
namespace inference::util {
    inline std::size_t file_size(std::ifstream& file) {
        file.seekg(0, std::ios::end);
        const auto size = file.tellg();
        file.seekg(0, std::ios::beg);
        return size;
    }

    inline std::vector<std::byte> read_file(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("failed to read " + path.string());
        }

        std::vector<std::byte> buffer(file_size(file));
        file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        return buffer;
    }

    inline void write_file(const std::filesystem::path& path, const std::uint8_t* raw_buffer, const std::size_t buffer_size) {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(raw_buffer), static_cast<std::streamsize>(buffer_size));
    }
} // namespace inference::util
