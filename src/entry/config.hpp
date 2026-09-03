#pragma once
#include "types/device.hpp"
#include <argparse/argparse.hpp>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

struct Args {
    std::filesystem::path model_path;
    std::optional<std::size_t> context_size;
    inference::types::Device device;

    [[nodiscard]] static Args parse_args(int argc, char* argv[]) {
        argparse::ArgumentParser parser("inference", "1.0", argparse::default_arguments::help);
        parser.add_description("LLM inference engine");
        parser.add_epilog("Example: inference --model ./model --device CPU --context-size 1024");

        parser.add_argument("--model").help("directory containing the model").required();
        parser.add_argument("--device").help("inference device").default_value(std::string("CPU")).choices("CPU", "CUDA");
        parser.add_argument("--context-size").help("maximum context size").scan<'u', std::size_t>();

        parser.parse_args(argc, argv);

        const auto model_path = std::filesystem::path{ parser.get<std::string>("--model") };
        if (!std::filesystem::exists(model_path)) {
            throw std::invalid_argument("model path does not exist: " + model_path.string());
        }

        const auto device = parser.get<std::string>("--device") == "CUDA" ? inference::types::Device::CUDA : inference::types::Device::CPU;

        return Args{
            .model_path = model_path,
            .context_size = parser.present<std::size_t>("--context-size"),
            .device = device,
        };
    }
};
