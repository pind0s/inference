#pragma once
#include "types/dtype.hpp"
#include "types/token.hpp"
#include <nlohmann/json.hpp>
#include <string>

namespace inference::model::qwen3 {
    struct Config {
        std::string model_type;
        types::TokenId eos_token_id{};
        std::size_t head_dim{};
        std::size_t hidden_size{};
        std::size_t intermediate_size{};
        std::size_t max_position_embeddings{};
        types::DType dtype{};
        std::size_t num_attention_heads{};
        std::size_t num_hidden_layers{};
        std::size_t num_key_value_heads{};
        float rms_norm_eps{};
        float rope_theta{};
        bool tie_word_embeddings{};
        std::size_t vocab_size{};
    };

    inline void from_json(const nlohmann::json& json, Config& config) {
        config = Config{
            .model_type = json.at("model_type").get<std::string>(),
            .eos_token_id = json.at("eos_token_id").get<types::TokenId>(),
            .head_dim = json.at("head_dim").get<std::size_t>(),
            .hidden_size = json.at("hidden_size").get<std::size_t>(),
            .intermediate_size = json.at("intermediate_size").get<std::size_t>(),
            .max_position_embeddings = json.at("max_position_embeddings").get<std::size_t>(),
            .dtype = types::dtype_from_string(json.at("torch_dtype").get<std::string>()),
            .num_attention_heads = json.at("num_attention_heads").get<std::size_t>(),
            .num_hidden_layers = json.at("num_hidden_layers").get<std::size_t>(),
            .num_key_value_heads = json.at("num_key_value_heads").get<std::size_t>(),
            .rms_norm_eps = json.at("rms_norm_eps").get<float>(),
            .rope_theta = json.at("rope_theta").get<float>(),
            .tie_word_embeddings = json.at("tie_word_embeddings").get<bool>(),
            .vocab_size = json.at("vocab_size").get<std::size_t>(),
        };
    }
} // namespace inference::model::qwen3
