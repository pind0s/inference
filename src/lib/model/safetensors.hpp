#pragma once
#include "backend/backend.hpp"
#include "model/weights.hpp"
#include "tensor/tensor.hpp"
#include "util/files.hpp"
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

namespace inference::safetensors {
    struct TensorDescriptor {
        std::size_t data_begin{};
        std::size_t data_end{};
        types::DType dtype{};
        TensorShape shape;

        [[nodiscard]] static TensorDescriptor from_json(const nlohmann::json& json) {
            std::array<std::size_t, 2> data_offsets{};
            json.at("data_offsets").get_to(data_offsets);
            return TensorDescriptor{
                .data_begin = data_offsets[0],
                .data_end = data_offsets[1],
                .dtype = types::dtype_from_string(json.at("dtype").get<std::string>()),
                .shape = TensorShape(json.at("shape").get<std::vector<std::size_t>>()),
            };
        }
    };

    // todo add support for models that are sharded into multiple files
    [[nodiscard]] inline Weights load_weights(const std::filesystem::path& path, Backend& backend) {
        constexpr std::string_view model_file_name = "model.safetensors";
        Weights weights;
        const auto file = util::read_file(path / model_file_name);
        const auto file_span = std::span{ file };

        std::uint64_t header_size{};
        std::memcpy(&header_size, file_span.data(), sizeof header_size);

        const auto header_bytes = file_span.subspan(sizeof header_size).first(header_size);
        const auto weight_bytes = file_span.subspan(sizeof header_size + header_size);

        for (const auto& [key, value] : nlohmann::json::parse(header_bytes).items()) {
            if (key == "__metadata__") {
                continue;
            }

            const auto [data_begin, data_end, dtype, shape] = TensorDescriptor::from_json(value);
            const auto source = weight_bytes.subspan(data_begin, data_end - data_begin);
            weights.insert(key, backend.make_tensor(source, shape, dtype));
        }

        return weights;
    }
} // namespace inference::safetensors
