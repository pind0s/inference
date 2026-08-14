#pragma once
#include "config.hpp"
#include "model/weights.hpp"
#include "ops/ops.hpp"
#include "runtime/context.hpp"
#include "tensor/tensor.hpp"

#include <memory>
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
    };

    struct MLP {
        Tensor gate_proj;
        Tensor up_proj;
        Tensor down_proj;
    };

    struct Layer {
        Tensor input_layernorm;
        SelfAttention self_attn;
        Tensor post_attention_layernorm;
        MLP mlp;
    };

    struct ScratchSpace {
        Tensor hidden_state;
        Tensor attention_block;
        Tensor query;
        Tensor key;
        Tensor value;
        Tensor normalized_query;
        Tensor normalized_key;
        Tensor attention_heads;
        Tensor projected_attention;
        Tensor mlp_block;
        Tensor gate;
        Tensor up;
        Tensor activated;
        Tensor mlp_output;
        Tensor logits;

        ScratchSpace(const Config& config, const types::DType dtype, const std::shared_ptr<allocator::Allocator>& allocator)
            : hidden_state{ Tensor::empty({ config.hidden_size }, dtype, allocator) },
              attention_block{ Tensor::empty({ config.hidden_size }, dtype, allocator) },
              query{ Tensor::empty({ config.num_attention_heads, config.head_dim }, dtype, allocator) },
              key{ Tensor::empty({ config.num_key_value_heads, config.head_dim }, dtype, allocator) },
              value{ Tensor::empty({ config.num_key_value_heads, config.head_dim }, dtype, allocator) },
              normalized_query{ Tensor::empty({ config.num_attention_heads, config.head_dim }, dtype, allocator) },
              normalized_key{ Tensor::empty({ config.num_key_value_heads, config.head_dim }, dtype, allocator) },
              attention_heads{ Tensor::empty({ config.num_attention_heads, config.head_dim }, dtype, allocator) },
              projected_attention{ Tensor::empty({ config.hidden_size }, dtype, allocator) },
              mlp_block{ Tensor::empty({ config.hidden_size }, dtype, allocator) },
              gate{ Tensor::empty({ config.intermediate_size }, dtype, allocator) },
              up{ Tensor::empty({ config.intermediate_size }, dtype, allocator) },
              activated{ Tensor::empty({ config.intermediate_size }, dtype, allocator) },
              mlp_output{ Tensor::empty({ config.hidden_size }, dtype, allocator) },
              logits{ Tensor::empty({ config.vocab_size }, dtype, allocator) } { }
    };

    struct Model {
        Config config;

        Tensor embed_tokens;
        std::vector<Layer> layers;
        Tensor norm;
        Tensor lm_head;
        ScratchSpace scratch;

        [[nodiscard]] static Model from_weights(const Config& config, Weights weights, Context& context) {
            const auto& allocator = context.cpu_context.allocator;
            auto model_weights = weights.scope("model");
            auto layer_weights = model_weights.scope("layers");

            auto embed_tokens = model_weights.take("embed_tokens.weight");
            auto final_norm = model_weights.take("norm.weight");
            auto lm_head = weights.take("lm_head.weight");

            if (config.tie_word_embeddings) {
                lm_head = embed_tokens;
            }

            std::vector<Layer> layers;
            layers.reserve(config.num_hidden_layers);
            for (std::size_t layer_index = 0; layer_index < config.num_hidden_layers; layer_index++) {
                auto layer = layer_weights.scope(layer_index);
                auto attention_weights = layer.scope("self_attn");
                auto mlp_weights = layer.scope("mlp");

                SelfAttention self_attn{
                    .q_proj = attention_weights.take("q_proj.weight"),
                    .k_proj = attention_weights.take("k_proj.weight"),
                    .v_proj = attention_weights.take("v_proj.weight"),
                    .o_proj = attention_weights.take("o_proj.weight"),
                    .q_norm = attention_weights.take("q_norm.weight"),
                    .k_norm = attention_weights.take("k_norm.weight"),
                };

                MLP mlp{
                    .gate_proj = mlp_weights.take("gate_proj.weight"),
                    .up_proj = mlp_weights.take("up_proj.weight"),
                    .down_proj = mlp_weights.take("down_proj.weight"),
                };

                layers.emplace_back(Layer{
                    .input_layernorm = layer.take("input_layernorm.weight"),
                    .self_attn = std::move(self_attn),
                    .post_attention_layernorm = layer.take("post_attention_layernorm.weight"),
                    .mlp = std::move(mlp),
                });
            }

            weights.expect_empty();
            const auto weights_dtype = embed_tokens.dtype();
            context.kv_cache = KVCache{ layers.size(), config.num_key_value_heads * config.head_dim, weights_dtype, allocator };

            return Model{
                .config = config,
                .embed_tokens = std::move(embed_tokens),
                .layers = std::move(layers),
                .norm = std::move(final_norm),
                .lm_head = std::move(lm_head),
                .scratch = ScratchSpace{ config, weights_dtype, allocator },
            };
        }

        // todo currently we do 0 optimizations for prefill, fine for now since makes the code a little simpler
        [[nodiscard]] Tensor forward(const std::size_t token_id, Context& context) {
            const auto token_pos = context.kv_cache.token_count;

            ops::embedding(token_id, embed_tokens, scratch.hidden_state);

            for (auto&& [layer, layer_cache] : std::views::zip(layers, context.kv_cache.layers)) {
                ops::rmsnorm(scratch.hidden_state, layer.input_layernorm, scratch.attention_block, config.rms_norm_eps);

                ops::matmul(scratch.attention_block, layer.self_attn.q_proj, scratch.query);

                ops::matmul(scratch.attention_block, layer.self_attn.k_proj, scratch.key);
                ops::matmul(scratch.attention_block, layer.self_attn.v_proj, scratch.value);

                ops::rmsnorm(scratch.query, layer.self_attn.q_norm, scratch.normalized_query, config.rms_norm_eps);
                ops::rmsnorm(scratch.key, layer.self_attn.k_norm, scratch.normalized_key, config.rms_norm_eps);

                ops::rope(scratch.normalized_query, config.num_attention_heads, config.head_dim, config.rope_theta, token_pos);
                ops::rope(scratch.normalized_key, config.num_key_value_heads, config.head_dim, config.rope_theta, context.kv_cache.token_count);

                ops::kv_cache_update(scratch.normalized_key, scratch.value, layer_cache.key, layer_cache.value, context.kv_cache.token_count);

                ops::self_attention(scratch.normalized_query, layer_cache.key, layer_cache.value, scratch.attention_heads, token_pos,
                                    config.num_attention_heads, config.num_key_value_heads, config.head_dim);

                ops::matmul(scratch.attention_heads, layer.self_attn.o_proj, scratch.projected_attention);
                ops::add(scratch.hidden_state, scratch.projected_attention, scratch.hidden_state);

                ops::rmsnorm(scratch.hidden_state, layer.post_attention_layernorm, scratch.mlp_block, config.rms_norm_eps);

                ops::matmul(scratch.mlp_block, layer.mlp.gate_proj, scratch.gate);
                ops::matmul(scratch.mlp_block, layer.mlp.up_proj, scratch.up);
                ops::silu_multiply(scratch.gate, scratch.up, scratch.activated);

                ops::matmul(scratch.activated, layer.mlp.down_proj, scratch.mlp_output);
                ops::add(scratch.hidden_state, scratch.mlp_output, scratch.hidden_state);
            }

            ops::rmsnorm(scratch.hidden_state, norm, scratch.attention_block, config.rms_norm_eps);
            ops::matmul(scratch.attention_block, lm_head, scratch.logits);
            ++context.kv_cache.token_count;
            return scratch.logits;
        }

        [[nodiscard]] Tensor prefill(std::span<const std::uint32_t> token_ids, Context& context) {
            auto logits = forward(token_ids.front(), context);
            for (const auto token_id : token_ids | std::views::drop(1)) {
                logits = forward(token_id, context);
            }
            return logits;
        }
    };

} // namespace inference::model::qwen3
