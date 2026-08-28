#pragma once
#include "backend/cpu/bf16.hpp"
#include "kv_cache/kv_cache.hpp"
#include "model/qwen3/qwen3.hpp"
#include <algorithm>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace test::util {
    struct FakeQwen3 {
        inference::model::qwen3::Model model;
        inference::KVCache kv_cache;
    };

    [[nodiscard]] inline FakeQwen3 make_fake_qwen3(inference::Backend& backend, const std::size_t context_capacity,
                                                   const std::size_t attention_head_count = 4, const std::size_t head_dim = 64) {
        using namespace inference;

        const auto hidden_size = attention_head_count * head_dim;
        const auto intermediate_size = hidden_size * 2;
        constexpr std::size_t key_value_head_count = 2;
        constexpr std::size_t layer_count = 2;
        constexpr std::size_t vocab_size = 1024;

        const auto config = model::qwen3::Config{
            .model_type = "qwen3",
            .head_dim = head_dim,
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
            .max_position_embeddings = context_capacity,
            .dtype = types::DType::BF16,
            .num_attention_heads = attention_head_count,
            .num_hidden_layers = layer_count,
            .num_key_value_heads = key_value_head_count,
            .rms_norm_eps = 1.0e-6F,
            .rope_theta = 1'000'000.0F,
            .vocab_size = vocab_size,
        };

        // todo slop
        auto random = std::mt19937{ 0 };
        auto distribution = std::uniform_real_distribution{ -1.0F, 1.0F };
        const auto make_tensor = [&](const TensorShape& shape) {
            auto values = std::vector<cpu::bf16_t>(shape.size());
            std::ranges::generate(values, [&] { return distribution(random); });

            return backend.make_tensor(std::as_bytes(std::span{ values }), shape, types::DType::BF16);
        };

        auto layers = std::vector<model::qwen3::Layer>{};
        layers.reserve(layer_count);
        for (std::size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
            auto self_attention = model::qwen3::SelfAttention{
                .q_proj = make_tensor({ attention_head_count * head_dim, hidden_size }),
                .k_proj = make_tensor({ key_value_head_count * head_dim, hidden_size }),
                .v_proj = make_tensor({ key_value_head_count * head_dim, hidden_size }),
                .o_proj = make_tensor({ hidden_size, attention_head_count * head_dim }),
                .q_norm = make_tensor({ head_dim }),
                .k_norm = make_tensor({ head_dim }),
            };

            auto mlp = model::qwen3::MLP{
                .gate_proj = make_tensor({ intermediate_size, hidden_size }),
                .up_proj = make_tensor({ intermediate_size, hidden_size }),
                .down_proj = make_tensor({ hidden_size, intermediate_size }),
            };

            auto layer = model::qwen3::Layer{
                .input_layernorm = make_tensor({ hidden_size }),
                .self_attn = std::move(self_attention),
                .post_attention_layernorm = make_tensor({ hidden_size }),
                .mlp = std::move(mlp),
            };
            layers.emplace_back(std::move(layer));
        }

        auto qwen = model::qwen3::Model{
            .config = config,
            .embed_tokens = make_tensor({ vocab_size, hidden_size }),
            .norm = make_tensor({ hidden_size }),
            .lm_head = make_tensor({ vocab_size, hidden_size }),
            .layers = std::move(layers),
            .rope_cache = make_rope_cache(context_capacity, head_dim, config.rope_theta, backend),
            .scratch = model::qwen3::ScratchSpace{ config, backend },
        };

        auto kv_cache = KVCache{ layer_count, key_value_head_count, head_dim, context_capacity, types::DType::BF16, backend };
        return { .model = std::move(qwen), .kv_cache = std::move(kv_cache) };
    }
} // namespace test::util
