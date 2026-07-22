#include <print>

#include "tokenizer/tokenizer.hpp"
#include "util/args.hpp"

int main(int argc, char* argv[]) try {
    const auto args = util::Args::parse_args(argc, argv);
    const auto tokenizer = inference::Tokenizer::parse_tokenizer(args.tokenizer_path, args.tokenizer_config_path);

    std::string prompt = "Hello world";
    auto tmp = tokenizer.pre_tokenize(prompt);

    for (const auto& token : tmp) {
        std::println("{}", token);
    }

    return 0;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return 1;
}
