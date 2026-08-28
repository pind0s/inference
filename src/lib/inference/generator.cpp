#include "generator.hpp"
#include "backend/backend.hpp"
#include "kv_cache/kv_cache.hpp"
#include "model/qwen3/qwen3.hpp"
#include "model/safetensors.hpp"
#include "tokenizer/tokenizer.hpp"
#include "types/device.hpp"
#include "util/files.hpp"
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace inference {
    Generator::Generator(Backend& backend, model::qwen3::Model model, tokenizer::Tokenizer tokenizer, KVCache kv_cache)
        : backend_(backend),
          model_(std::move(model)),
          tokenizer_(std::move(tokenizer)),
          kv_cache_(std::move(kv_cache)) { }

    Generator Generator::load(const std::filesystem::path& model_path, types::Device device, std::optional<std::size_t> max_context_size) {
        auto& backend = Backend::select_backend(device);
        const auto config = load_config(model_path);
        const auto context_size = max_context_size.value_or(config.max_position_embeddings);
        auto weights = safetensors::load_weights(model_path, backend);
        auto kv_cache = KVCache(config.num_hidden_layers, config.num_key_value_heads, config.head_dim, context_size, config.dtype, backend);
        auto tokenizer = tokenizer::Tokenizer(model_path);
        auto model = model::qwen3::Model::build(config, std::move(weights), context_size, backend);
        return Generator{ backend, std::move(model), std::move(tokenizer), std::move(kv_cache) };
    }

    std::string Generator::generate(std::string prompt) {
        kv_cache_.reset();

        prompt = model_.apply_chat_template(prompt);
        const auto input_tokens = tokenizer_.tokenize(prompt);
        if (input_tokens.size() >= kv_cache_.capacity) {
            throw std::length_error("prompt does not fit in the model context");
        }

        tokenizer::TokenList response;
        auto next_token = model_.prefill(input_tokens, backend_, kv_cache_);
        while (kv_cache_.token_count + 1 < kv_cache_.capacity) {
            next_token = model_.forward(next_token, backend_, kv_cache_);

            if (next_token == model_.config.eos_token_id) {
                break;
            }
            response.emplace_back(next_token);
        }

        return tokenizer_.decode(response);
    }

    model::qwen3::Config Generator::load_config(const std::filesystem::path& model_path) {
        constexpr std::string_view config_file = "config.json";
        auto file = util::read_file(model_path / config_file);
        return nlohmann::json::parse(file).get<model::qwen3::Config>();
    }
} // namespace inference
