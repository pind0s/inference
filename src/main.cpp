#include <print>

#include "allocator/cpu_allocator.hpp"
#include "storage/storage.hpp"
#include "tensor/tensor.hpp"
#include "tokenizer/tokenizer.hpp"
#include "util/args.hpp"

int main(int argc, char* argv[]) try {
    const auto args = util::Args::parse_args(argc, argv);
    const auto tokenizer = inference::Tokenizer::load_tokenizer(args.tokenizer_path, args.tokenizer_config_path);

    std::string prompt = "hello world";
    auto tokens = tokenizer.tokenize(prompt);
    std::println("{}", tokenizer.decode(tokens));

    return 0;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return 1;
}
