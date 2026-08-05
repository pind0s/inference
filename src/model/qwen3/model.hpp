#pragma once
#include <utility>
#include <vector>

#include "config.hpp"
#include "model/weights.hpp"
#include "tensor/tensor.hpp"

namespace inference::model {
    namespace qwen3 {
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
    } // namespace qwen3


    // todo just rename this to model and put in qwen3 namespace?
    struct Qwen3Model {
        Qwen3Config config;

        Tensor embed_tokens;
        std::vector<qwen3::Layer> layers;
        Tensor norm;
        Tensor lm_head;

        [[nodiscard]] static Qwen3Model from_weights(const Qwen3Config& config, Weights weights) {
            auto model_weights = weights.scope("model");
            auto layer_weights = model_weights.scope("layers");

            auto embed_tokens = model_weights.take("embed_tokens.weight");
            auto final_norm = model_weights.take("norm.weight");
            auto lm_head = weights.take("lm_head.weight");

            if (config.tie_word_embeddings) {
                lm_head = Tensor::from_storage(embed_tokens.storage(), embed_tokens.shape(), embed_tokens.dtype());
            }

            std::vector<qwen3::Layer> layers;
            layers.reserve(config.num_hidden_layers);
            for (std::size_t layer_index = 0; layer_index < config.num_hidden_layers; layer_index++) {
                auto layer = layer_weights.scope(layer_index);
                auto attention_weights = layer.scope("self_attn");
                auto mlp_weights = layer.scope("mlp");

                qwen3::SelfAttention self_attn{
                    .q_proj = attention_weights.take("q_proj.weight"),
                    .k_proj = attention_weights.take("k_proj.weight"),
                    .v_proj = attention_weights.take("v_proj.weight"),
                    .o_proj = attention_weights.take("o_proj.weight"),
                    .q_norm = attention_weights.take("q_norm.weight"),
                    .k_norm = attention_weights.take("k_norm.weight"),
                };

                qwen3::MLP mlp{
                    .gate_proj = mlp_weights.take("gate_proj.weight"),
                    .up_proj = mlp_weights.take("up_proj.weight"),
                    .down_proj = mlp_weights.take("down_proj.weight"),
                };

                layers.emplace_back(qwen3::Layer{
                    .input_layernorm = layer.take("input_layernorm.weight"),
                    .self_attn = std::move(self_attn),
                    .post_attention_layernorm = layer.take("post_attention_layernorm.weight"),
                    .mlp = std::move(mlp),
                });
            }

            weights.expect_empty();

            return Qwen3Model{
                .config = config,
                .embed_tokens = std::move(embed_tokens),
                .layers = std::move(layers),
                .norm = std::move(final_norm),
                .lm_head = std::move(lm_head),
            };
        }
    };

} // namespace inference::model
