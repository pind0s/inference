#pragma once
#include "util/files.hpp"
#include <argparse/argparse.hpp>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>

namespace util {
    struct Args {
        std::filesystem::path model_path;
        std::filesystem::path tokenizer_path;
        std::filesystem::path tokenizer_config_path;
        std::filesystem::path model_config_path;
        std::optional<std::size_t> context_size;

        [[nodiscard]] static Args parse_args(const int argc, char* argv[]) {
            argparse::ArgumentParser parser{ "inference engine", "1.0" };
            parser.add_description("Inference engine");
            parser.add_epilog("TODO epilog"); // todo
            parser.add_argument("-m", "--model").help("path that contains the model").required();
            parser.add_argument("--context-size").help("maximum sequence length in tokens").scan<'u', std::size_t>();
            parser.parse_args(argc, argv);

            const auto model_path = std::filesystem::path(parser.get<std::string>("model"));

            return Args{
                .model_path = model_path,
                .tokenizer_path = require_file(model_path / "tokenizer.json"),
                .tokenizer_config_path = require_file(model_path / "tokenizer_config.json"),
                .model_config_path = require_file(model_path / "config.json"),
                .context_size = parser.present<std::size_t>("--context-size"),
            };
        }
    };
} // namespace util
