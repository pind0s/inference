#include "config.hpp"
#include "inference/generator.hpp"
#include <exception>
#include <print>
#include <string>

int main(const int argc, char* argv[]) try {
    const auto [model_path, context_size, device] = Args::parse_args(argc, argv);
    const std::string prompt = "Tell me a short story about LLMs";
    std::println("Prompt: {}", prompt);

    auto generator = inference::Generator::load(model_path, device, context_size);
    const auto response = generator.generate(prompt);

    std::println("Response: {}", response);
} catch (const std::exception& error) {
    std::println(stderr, "Error: {}", error.what());
    return 1;
}
