#include "model/qwen3/qwen3.hpp"
#include "model/safetensors.hpp"
#include "tokenizer/tokenizer.hpp"
#include "util/args.hpp"
#include <chrono>
#include <format>
#include <limits>
#include <print>
#include <string>
#include <utility>

std::string apply_chat_template(const std::string& prompt) {
    return std::format("<|im_start|>user\n{}<|im_end|>\n"
                       "<|im_start|>assistant\n"
                       "<think>\n\n"
                       "</think>",
                       prompt);
}

int main(int argc, char* argv[]) try {
    using namespace inference;

    const auto args = util::Args::parse_args(argc, argv);

    auto runtime_context = Context();

    const auto json_config = nlohmann::json::parse(util::read_file(args.model_config_path).value());
    const auto config = model::qwen3::Config::from_json(json_config);

    auto weights = safetensors::load_weights_from_dir(args.model_path, runtime_context.cpu_context.allocator);
    auto model = model::qwen3::Model::from_weights(config, std::move(weights), runtime_context);
    const auto tokenizer = Tokenizer::load_tokenizer(args.tokenizer_path, args.tokenizer_config_path);

    const std::string prompt = "Write a 200 word essay about how llms work";
    const auto input_tokens = tokenizer.tokenize(apply_chat_template(prompt));
    TokenList response_tokens;

    constexpr std::size_t MAX_NEW_TOKENS = 200;

    auto logits = model.prefill(input_tokens, runtime_context);
    const auto generation_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < MAX_NEW_TOKENS; ++i) {
        auto next_token = ops::argmax(logits);

        if (next_token == 151645 || next_token == 151643) [[unlikely]] {
            break;
        }

        response_tokens.emplace_back(next_token);

        logits = model.forward(next_token, runtime_context);
    }
    const auto generation_elapsed = std::chrono::steady_clock::now() - generation_start;
    const auto generation_seconds = std::chrono::duration<double>(generation_elapsed).count();

    std::println("Prompt: {}", prompt);
    std::println("Response: {}", tokenizer.decode(response_tokens));
    std::println("Generation speed: {:.2f} tokens/s ({} tokens in {:.2f} s)", generation_seconds > 0.0 ? response_tokens.size() / generation_seconds : 0.0,
                 response_tokens.size(), generation_seconds);

    return 0;
} catch (const std::exception& e) {
    std::println(stderr, "Error: {}", e.what());
    return 1;
}
