#pragma once
#include <argparse/argparse.hpp>
#include <filesystem>
#include <string>

#include "util/files.hpp"

namespace util {
    struct Args {
        std::filesystem::path model_path;
        std::filesystem::path tokenizer_path;
        std::filesystem::path tokenizer_config_path;

        [[nodiscard]] static Args parse_args(const int argc, char* argv[]) {
            argparse::ArgumentParser parser{"inference engine", "1.0"};
            parser.add_description("Inference engine");
            parser.add_epilog("TODO epilog");
            parser.add_argument("-m", "--model").help("path that contains the model").required();
            parser.parse_args(argc, argv);

            const auto model_path = std::filesystem::path(parser.get<std::string>("model"));

            return Args{
                .model_path = model_path,
                .tokenizer_path = require_file(model_path / "tokenizer.json"),
                .tokenizer_config_path = require_file(model_path / "tokenizer_config.json"),
            };
        }
    };
} // namespace util
