#pragma once
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "config.hpp"
#include "model/weights.hpp"
#include "ops/ops.hpp"
#include "runtime/context.hpp"
#include "tensor/tensor.hpp"

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
        Tensor attention;
        Tensor projected_attention;
        Tensor mlp_block;
        Tensor gate;
        Tensor up;
        Tensor activated;
        Tensor mlp_output;

        // todo I think we can simplify this
        ScratchSpace(const Config& config, const std::size_t sequence_size, const types::DType dtype,
                     const std::shared_ptr<allocator::BaseAllocator>& allocator)
            : hidden_state{Tensor::empty({sequence_size, config.hidden_size}, dtype, allocator)},
              attention_block{Tensor::empty({sequence_size, config.hidden_size}, dtype, allocator)},
              query{Tensor::empty({sequence_size, config.num_attention_heads * config.head_dim}, dtype, allocator)},
              key{Tensor::empty({sequence_size, config.num_key_value_heads * config.head_dim}, dtype, allocator)},
              value{Tensor::empty({sequence_size, config.num_key_value_heads * config.head_dim}, dtype, allocator)},
              normalized_query{Tensor::empty({sequence_size, config.num_attention_heads, config.head_dim}, dtype, allocator)},
              normalized_key{Tensor::empty({sequence_size, config.num_key_value_heads, config.head_dim}, dtype, allocator)},
              attention_heads{Tensor::empty({sequence_size, config.num_attention_heads, config.head_dim}, dtype, allocator)},
              attention{attention_heads.reshape({sequence_size, config.num_attention_heads * config.head_dim})},
              projected_attention{Tensor::empty({sequence_size, config.hidden_size}, dtype, allocator)},
              mlp_block{Tensor::empty({sequence_size, config.hidden_size}, dtype, allocator)},
              gate{Tensor::empty({sequence_size, config.intermediate_size}, dtype, allocator)},
              up{Tensor::empty({sequence_size, config.intermediate_size}, dtype, allocator)},
              activated{Tensor::empty({sequence_size, config.intermediate_size}, dtype, allocator)},
              mlp_output{Tensor::empty({sequence_size, config.hidden_size}, dtype, allocator)} { }
    };

    struct Model {
        Config config;

        Tensor embed_tokens;
        std::vector<Layer> layers;
        Tensor norm;
        Tensor lm_head;

        [[nodiscard]] static Model from_weights(const Config& config, Weights weights) {
            auto model_weights = weights.scope("model");
            auto layer_weights = model_weights.scope("layers");

            auto embed_tokens = model_weights.take("embed_tokens.weight");
            auto final_norm = model_weights.take("norm.weight");
            auto lm_head = weights.take("lm_head.weight");

            if (config.tie_word_embeddings) {
                lm_head = Tensor::from_storage(embed_tokens.storage(), embed_tokens.shape(), embed_tokens.dtype());
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

            return Model{
                .config = config,
                .embed_tokens = std::move(embed_tokens),
                .layers = std::move(layers),
                .norm = std::move(final_norm),
                .lm_head = std::move(lm_head),
            };
        }

        [[nodiscard]] Tensor forward(const Tensor& input_ids, Context& context) const {
            const auto sequence_size = input_ids.dim(0);

            // todo
            if (sequence_size > 1024) {
                throw std::invalid_argument("can't generate more than 1024 tokens.");
            }

            const auto allocator = context.cpu_context.allocator;
            const auto weights_dtype = embed_tokens.dtype();
            auto scratch = ScratchSpace{config, sequence_size, weights_dtype, allocator};

            const auto query_size = config.num_attention_heads * config.head_dim;
            const auto key_value_size = config.num_key_value_heads * config.head_dim;

            auto logits = Tensor::empty({sequence_size, config.vocab_size}, weights_dtype, allocator);
            ops::embedding(input_ids, embed_tokens, scratch.hidden_state, sequence_size, config.hidden_size);

            for (const auto& layer : layers) {
                ops::rmsnorm(scratch.hidden_state, layer.input_layernorm, scratch.attention_block, sequence_size, config.hidden_size,
                             config.rms_norm_eps);

                ops::matmul(scratch.attention_block, layer.self_attn.q_proj, scratch.query, sequence_size, query_size, config.hidden_size);

                ops::matmul(scratch.attention_block, layer.self_attn.k_proj, scratch.key, sequence_size, key_value_size, config.hidden_size);
                ops::matmul(scratch.attention_block, layer.self_attn.v_proj, scratch.value, sequence_size, key_value_size, config.hidden_size);

                ops::qk_norm(scratch.query, layer.self_attn.q_norm, scratch.normalized_query, sequence_size * config.num_attention_heads,
                             config.head_dim, config.rms_norm_eps);
                ops::qk_norm(scratch.key, layer.self_attn.k_norm, scratch.normalized_key, sequence_size * config.num_key_value_heads,
                             config.head_dim, config.rms_norm_eps);
                ops::rope(scratch.normalized_query, sequence_size, config.num_attention_heads, config.head_dim, config.rope_theta);
                ops::rope(scratch.normalized_key, sequence_size, config.num_key_value_heads, config.head_dim, config.rope_theta);

                ops::self_attention(scratch.normalized_query, scratch.normalized_key, scratch.value, scratch.attention_heads, sequence_size,
                                    config.num_attention_heads, config.num_key_value_heads, config.head_dim);

                ops::matmul(scratch.attention, layer.self_attn.o_proj, scratch.projected_attention, sequence_size, config.hidden_size,
                            query_size);
                ops::add(scratch.hidden_state, scratch.projected_attention, scratch.attention_block, sequence_size * config.hidden_size);
                std::swap(scratch.hidden_state, scratch.attention_block);

                ops::rmsnorm(scratch.hidden_state, layer.post_attention_layernorm, scratch.mlp_block, sequence_size, config.hidden_size,
                             config.rms_norm_eps);

                ops::matmul(scratch.mlp_block, layer.mlp.gate_proj, scratch.gate, sequence_size, config.intermediate_size, config.hidden_size);
                ops::matmul(scratch.mlp_block, layer.mlp.up_proj, scratch.up, sequence_size, config.intermediate_size, config.hidden_size);
                ops::silu_multiply(scratch.gate, scratch.up, scratch.activated, sequence_size * config.intermediate_size);

                ops::matmul(scratch.activated, layer.mlp.down_proj, scratch.mlp_output, sequence_size, config.hidden_size,
                            config.intermediate_size);
                ops::add(scratch.hidden_state, scratch.mlp_output, scratch.mlp_block, sequence_size * config.hidden_size);
                std::swap(scratch.hidden_state, scratch.mlp_block);
            }

            ops::rmsnorm(scratch.hidden_state, norm, scratch.attention_block, sequence_size, config.hidden_size, config.rms_norm_eps);
            ops::matmul(scratch.attention_block, lm_head, logits, sequence_size, config.vocab_size, config.hidden_size);
            return logits;
        }
    };

} // namespace inference::model::qwen3
