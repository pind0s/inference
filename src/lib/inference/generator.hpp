#pragma once
#include "backend/backend.hpp"
#include "kv_cache/kv_cache.hpp"
#include "model/qwen3/qwen3.hpp"
#include "tokenizer/tokenizer.hpp"
#include <filesystem>
#include <optional>
#include <string>

namespace inference {
    class Generator : public util::MoveOnly {
    public:
        [[nodiscard]] static Generator load(const std::filesystem::path& model_path, types::Device device,
                                            std::optional<std::size_t> max_context_size = std::nullopt);

        [[nodiscard]] std::string generate(std::string prompt);

    private:
        Generator(Backend& backend, model::qwen3::Model model, tokenizer::Tokenizer tokenizer, KVCache kv_cache);

        [[nodiscard]] static model::qwen3::Config load_config(const std::filesystem::path& model_path);

        Backend& backend_;
        model::qwen3::Model model_;
        tokenizer::Tokenizer tokenizer_;
        KVCache kv_cache_;
    };
} // namespace inference
