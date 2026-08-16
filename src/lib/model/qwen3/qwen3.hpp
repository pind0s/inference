#pragma once
#include "backend/rope_cache.hpp"
#include "config.hpp"
#include "kv_cache/kv_cache.hpp"
#include "model/weights.hpp"
#include "tensor/tensor.hpp"
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace inference::model::qwen3 {

    struct SelfAttention {
        Tensor q_proj;
        Tensor k_proj;
        Tensor v_proj;
        Tensor o_proj;

        Tensor q_norm;
        Tensor k_norm;

        [[nodiscard]] static SelfAttention load(const Weights& layer_weights) {
            auto weights = layer_weights.scope("self_attn");
            return {
                .q_proj = weights.take("q_proj.weight"),
                .k_proj = weights.take("k_proj.weight"),
                .v_proj = weights.take("v_proj.weight"),
                .o_proj = weights.take("o_proj.weight"),
                .q_norm = weights.take("q_norm.weight"),
                .k_norm = weights.take("k_norm.weight"),
            };
        }
    };

    struct MLP {
        Tensor gate_proj;
        Tensor up_proj;
        Tensor down_proj;

        [[nodiscard]] static MLP load(const Weights& layer_weights) {
            auto weights = layer_weights.scope("mlp");
            return {
                .gate_proj = weights.take("gate_proj.weight"),
                .up_proj = weights.take("up_proj.weight"),
                .down_proj = weights.take("down_proj.weight"),
            };
        }
    };

    struct Layer {
        Tensor input_layernorm;
        SelfAttention self_attn;
        Tensor post_attention_layernorm;
        MLP mlp;

        [[nodiscard]] static Layer load(Weights weights) {
            return {
                .input_layernorm = weights.take("input_layernorm.weight"),
                .self_attn = SelfAttention::load(weights),
                .post_attention_layernorm = weights.take("post_attention_layernorm.weight"),
                .mlp = MLP::load(weights),
            };
        }
    };

    struct ScratchSpace {
        Tensor hidden_state;
        Tensor hidden_scratch;
        Tensor query;
        Tensor key;
        Tensor value;
        Tensor gate;
        Tensor up;
        Tensor logits;

        ScratchSpace(const Config& config, Backend& backend)
            : hidden_state(Tensor::empty({ config.hidden_size }, config.dtype, backend)),
              hidden_scratch(Tensor::empty({ config.hidden_size }, config.dtype, backend)),
              query(Tensor::empty({ config.num_attention_heads, config.head_dim }, config.dtype, backend)),
              key(Tensor::empty({ config.num_key_value_heads, config.head_dim }, config.dtype, backend)),
              value(Tensor::empty({ config.num_key_value_heads, config.head_dim }, config.dtype, backend)),
              gate(Tensor::empty({ config.intermediate_size }, config.dtype, backend)),
              up(Tensor::empty({ config.intermediate_size }, config.dtype, backend)),
              logits(Tensor::empty({ config.vocab_size }, config.dtype, backend)) { }
    };

    struct Model {
        Config config;
        Tensor embed_tokens;
        Tensor norm;
        Tensor lm_head;
        std::vector<Layer> layers;
        RopeCache rope_cache;
        ScratchSpace scratch;

        [[nodiscard]] static Model build(const Config& config, Weights weights, const std::size_t context_size, Backend& backend) {
            auto model_weights = weights.scope("model");
            auto layer_weights = model_weights.scope("layers");

            std::vector<Layer> layers;
            layers.reserve(config.num_hidden_layers);
            for (std::size_t layer_index = 0; layer_index < config.num_hidden_layers; layer_index++) {
                layers.push_back(Layer::load(layer_weights.scope(layer_index)));
            }

            auto embed_tokens = model_weights.take("embed_tokens.weight");
            auto norm = model_weights.take("norm.weight");
            auto lm_head = config.tie_word_embeddings ? embed_tokens : weights.take("lm_head.weight");

            return Model{
                .config = config,
                .embed_tokens = std::move(embed_tokens),
                .norm = std::move(norm),
                .lm_head = std::move(lm_head),
                .layers = std::move(layers),
                .rope_cache = make_rope_cache(context_size, config.head_dim, config.rope_theta, backend),
                .scratch = ScratchSpace{ config, backend },
            };
        }

        // todo currently we do 0 optimizations for prefill, fine for now since makes the code a little simpler
        [[nodiscard]] types::TokenId forward(const types::TokenId token_id, Backend& backend, KVCache& kv_cache) {
            const auto token_pos = kv_cache.token_count;

            backend.embedding(token_id, embed_tokens, scratch.hidden_state);

            for (auto&& [layer, layer_cache] : std::views::zip(layers, kv_cache.layers)) {
                backend.rmsnorm(scratch.hidden_state, layer.input_layernorm, scratch.hidden_scratch, config.rms_norm_eps);

                backend.matmul(scratch.hidden_scratch, layer.self_attn.q_proj, scratch.query);
                backend.matmul(scratch.hidden_scratch, layer.self_attn.k_proj, scratch.key);
                backend.matmul(scratch.hidden_scratch, layer.self_attn.v_proj, scratch.value);

                backend.rmsnorm(scratch.query, layer.self_attn.q_norm, scratch.query, config.rms_norm_eps);
                backend.rmsnorm(scratch.key, layer.self_attn.k_norm, scratch.key, config.rms_norm_eps);

                backend.rope(scratch.query, rope_cache, token_pos);
                backend.rope(scratch.key, rope_cache, token_pos);

                backend.kv_cache_update(scratch.key, scratch.value, layer_cache.key, layer_cache.value, kv_cache.token_count);

                backend.self_attention(scratch.query, layer_cache.key, layer_cache.value, scratch.query, token_pos);

                backend.matmul(scratch.query, layer.self_attn.o_proj, scratch.hidden_scratch);
                backend.add(scratch.hidden_state, scratch.hidden_scratch, scratch.hidden_state);

                backend.rmsnorm(scratch.hidden_state, layer.post_attention_layernorm, scratch.hidden_scratch, config.rms_norm_eps);

                backend.matmul(scratch.hidden_scratch, layer.mlp.gate_proj, scratch.gate);
                backend.matmul(scratch.hidden_scratch, layer.mlp.up_proj, scratch.up);
                backend.silu(scratch.gate, scratch.up, scratch.gate);

                backend.matmul(scratch.gate, layer.mlp.down_proj, scratch.hidden_scratch);
                backend.add(scratch.hidden_state, scratch.hidden_scratch, scratch.hidden_state);
            }

            backend.rmsnorm(scratch.hidden_state, norm, scratch.hidden_scratch, config.rms_norm_eps);
            backend.matmul(scratch.hidden_scratch, lm_head, scratch.logits);
            ++kv_cache.token_count;
            return backend.argmax(scratch.logits);
        }

        [[nodiscard]] types::TokenId prefill(const std::span<const types::TokenId> token_ids, Backend& backend, KVCache& kv_cache) {
            for (const auto token_id : token_ids.first(token_ids.size() - 1)) {
                auto _ = forward(token_id, backend, kv_cache);
            }
            return token_ids.back();
        }
    };
} // namespace inference::model::qwen3
