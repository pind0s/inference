#pragma once
#include "allocator/allocator.hpp"
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
    // todo add support for manual mapped files
    [[nodiscard]] inline Weights load_weights_from_dir(const std::filesystem::path& path, const std::shared_ptr<allocator::Allocator>& allocator) {
        Weights weights;

        const auto file = util::read_file(util::require_file(path / "model.safetensors")).value();
        const auto file_bytes = std::span{ file };

        std::uint64_t header_size{};
        std::memcpy(&header_size, file_bytes.data(), sizeof header_size);

        const auto header_bytes = file_bytes.subspan(sizeof header_size).first(header_size);
        const auto weight_bytes = file_bytes.subspan(sizeof header_size + header_size);
        const auto metadata = nlohmann::json::parse(header_bytes);

        for (const auto& [key, value] : metadata.items()) {
            if (key == "__metadata__") {
                continue;
            }

            const auto descriptor = TensorDescriptor::from_json(value);
            auto tensor = Tensor::empty(descriptor.shape, descriptor.dtype, allocator);
            auto source = weight_bytes.subspan(descriptor.data_begin, tensor.size_bytes());
            std::ranges::uninitialized_copy(source, tensor.as_writable_bytes());
            // todo this won't work with cuda. maybe we should always first
            //  copy to cpu and then transfer tensor to cuda? or we can
            //  create op that will copy depending on the device
            weights.insert(key, std::move(tensor));
        }

        return weights;
    }
} // namespace inference::safetensors
