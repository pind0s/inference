#include "inference/generator.hpp"
#include <exception>
#include <print>
#include <string>

int main(const int argc, char* argv[]) try {
    if (argc < 2) throw std::invalid_argument("entry: missing model path");

    const std::string prompt = "Tell me a short story about llms";
    constexpr std::size_t max_context_size = 1024;
    constexpr auto device = inference::types::Device::CUDA;

    auto generator = inference::Generator::load(argv[1], device, max_context_size);

    std::println("Prompt: {}", prompt);
    std::println("Response: {}", generator.generate(prompt));
} catch (const std::exception& error) {
    std::println(stderr, "Error: {}", error.what());
    return 1;
}
