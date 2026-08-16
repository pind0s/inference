#include "inference/generator.hpp"
#include <chrono>
#include <exception>
#include <format>
#include <print>
#include <string>
#include <string_view>

namespace {
    [[nodiscard]] std::string apply_chat_template(const std::string_view prompt) {
        return std::format("<|im_start|>user\n{}<|im_end|>\n"
                           "<|im_start|>assistant\n"
                           "<think>\n\n"
                           "</think>",
                           prompt);
    }

    constexpr std::size_t max_context_size = 1024;
    constexpr std::string_view prompt = "Tell me 3 jokes about llms";
} // namespace

int main(const int argc, char* argv[]) try {

    if (argc < 2) {
        throw std::invalid_argument("missing model path");
    }

    std::println("Prompt: {}\n", prompt);

    auto generator = inference::Generator::load(argv[1], inference::types::Device::CPU, max_context_size);
    const auto start = std::chrono::steady_clock::now();
    const auto completion = generator.generate(apply_chat_template(prompt));
    const auto elapsed = std::chrono::steady_clock::now() - start;
    std::println("Completion: {}\n", completion);
    std::println("Generation time: {} seconds", std::chrono::duration<double>(elapsed).count());
    return 0;
} catch (const std::exception& error) {
    std::println(stderr, "Error: {}", error.what());
    return 1;
}
