#pragma once
#include "types/activation.hpp"
#include <nlohmann/json.hpp>
#include <string>

namespace inference::model::qwen3 {
    struct Config {
        std::string model_type;
        std::size_t eos_token_id{};
        std::size_t head_dim{};
        types::Activation hidden_act{};
        std::size_t hidden_size{};
        std::size_t intermediate_size{};
        std::size_t max_position_embeddings{};
        std::size_t max_window_layers{};
        std::size_t num_attention_heads{};
        std::size_t num_hidden_layers{};
        std::size_t num_key_value_heads{};
        float rms_norm_eps{};
        float rope_theta{};
        bool tie_word_embeddings{};
        std::optional<std::size_t> sliding_window;
        bool use_sliding_window{};
        std::size_t vocab_size{};

        [[nodiscard]] static Config from_json(const nlohmann::json& json) {
            Config config;
            json.at("model_type").get_to(config.model_type);
            json.at("eos_token_id").get_to(config.eos_token_id);
            json.at("head_dim").get_to(config.head_dim);
            json.at("hidden_size").get_to(config.hidden_size);
            json.at("intermediate_size").get_to(config.intermediate_size);
            json.at("max_position_embeddings").get_to(config.max_position_embeddings);
            json.at("max_window_layers").get_to(config.max_window_layers);
            json.at("num_attention_heads").get_to(config.num_attention_heads);
            json.at("num_hidden_layers").get_to(config.num_hidden_layers);
            json.at("num_key_value_heads").get_to(config.num_key_value_heads);
            json.at("rms_norm_eps").get_to(config.rms_norm_eps);
            json.at("rope_theta").get_to(config.rope_theta);
            json.at("tie_word_embeddings").get_to(config.tie_word_embeddings);
            json.at("sliding_window").get_to(config.sliding_window);
            json.at("use_sliding_window").get_to(config.use_sliding_window);
            json.at("vocab_size").get_to(config.vocab_size);
            config.hidden_act = types::activation_from_string(json.at("hidden_act").get<std::string>());
            return config;
        }
    };

} // namespace inference::model::qwen3
