#pragma once
#include "tensor/tensor_view.hpp"
#include <cuda/cmath>
#include <cuda/launch>
#include <cuda/std/algorithm>
#include <cuda_bf16.h>

namespace inference::gpu::kernels {
    namespace detail {
        constexpr std::size_t KV_TILE_SIZE = 32;
        [[nodiscard]] constexpr std::size_t shared_memory_element_count(const std::size_t head_size) noexcept {
            return (2 * KV_TILE_SIZE + 2) * head_size + KV_TILE_SIZE + 2;
        }

        struct FlashAttentionBf16 {
            struct Args {
                TensorView<const __nv_bfloat16> query;
                TensorView<const __nv_bfloat16> key;
                TensorView<const __nv_bfloat16> value;
                TensorView<__nv_bfloat16> output;
                std::size_t token_pos;
            };

            // todo add cub reduction, and i think we can simplify the code?
            template <typename Configuration>
            __device__ void operator()(const Configuration& config, Args args) const {
                const auto head_dimension_index = cuda::gpu_thread.rank(cuda::block, config);
                const auto query_head_index = cuda::block.rank(cuda::grid, config);
                const auto q_head_count = args.query.dim(0);
                const auto kv_head_count = args.key.dim(1);
                const auto head_size = args.query.dim(1);
                const auto query_heads_per_kv_head = q_head_count / kv_head_count;
                const auto kv_head_index = query_head_index / query_heads_per_kv_head;
                const auto token_count = args.token_pos + 1;
                const auto attention_scale = 1.0F / cuda::std::sqrtf(static_cast<float>(head_size));

                // todo look if we can cast shared_memory to cuda::std::span or struct of spans? this sucks rn
                auto shared_memory = cuda::dynamic_shared_memory(config);
                auto* q_shared = shared_memory.data();
                auto* k_shared = q_shared + head_size;
                auto* v_shared = k_shared + KV_TILE_SIZE * head_size;
                auto* reduction = v_shared + KV_TILE_SIZE * head_size;
                auto* scores_shared = reduction + head_size;
                auto* tile_maximum_shared = scores_shared + KV_TILE_SIZE;
                auto* tile_exponential_sum_shared = tile_maximum_shared + 1;

                q_shared[head_dimension_index] = __bfloat162float(args.query(query_head_index, head_dimension_index));
                __syncthreads();

                float running_maximum = -cuda::std::numeric_limits<float>::infinity();
                float running_denominator = 0.0F;
                float running_numerator = 0.0F;

                for (std::size_t tile_start = 0; tile_start < token_count; tile_start += KV_TILE_SIZE) {
                    const auto tile_token_count = token_count - tile_start < KV_TILE_SIZE ? token_count - tile_start : KV_TILE_SIZE;

                    for (std::size_t token_in_tile = 0; token_in_tile < tile_token_count; ++token_in_tile) {
                        const auto token_index = tile_start + token_in_tile;
                        const auto shared_offset = token_in_tile * head_size + head_dimension_index;

                        k_shared[shared_offset] = __bfloat162float(args.key(token_index, kv_head_index, head_dimension_index));
                        v_shared[shared_offset] = __bfloat162float(args.value(token_index, kv_head_index, head_dimension_index));
                    }
                    __syncthreads();

                    for (std::size_t token_in_tile = 0; token_in_tile < tile_token_count; ++token_in_tile) {
                        const auto shared_offset = token_in_tile * head_size + head_dimension_index;
                        reduction[head_dimension_index] = q_shared[head_dimension_index] * k_shared[shared_offset];
                        __syncthreads();

                        for (std::size_t active_count = head_size; active_count > 1; active_count = (active_count + 1) / 2) {
                            const auto upper_half_start = (active_count + 1) / 2;
                            if (head_dimension_index < upper_half_start && head_dimension_index + upper_half_start < active_count) {
                                reduction[head_dimension_index] += reduction[head_dimension_index + upper_half_start];
                            }
                            __syncthreads();
                        }

                        if (head_dimension_index == 0) {
                            scores_shared[token_in_tile] = reduction[0] * attention_scale;
                        }
                        __syncthreads();
                    }

                    if (head_dimension_index == 0) {
                        auto tile_maximum = -cuda::std::numeric_limits<float>::infinity();
                        for (std::size_t token_in_tile = 0; token_in_tile < tile_token_count; ++token_in_tile) {
                            tile_maximum = cuda::std::max(tile_maximum, scores_shared[token_in_tile]);
                        }

                        float tile_exponential_sum = 0.0F;
                        for (std::size_t token_in_tile = 0; token_in_tile < tile_token_count; ++token_in_tile) {
                            const auto weight = cuda::std::exp(scores_shared[token_in_tile] - tile_maximum);
                            scores_shared[token_in_tile] = weight;
                            tile_exponential_sum += weight;
                        }

                        *tile_maximum_shared = tile_maximum;
                        *tile_exponential_sum_shared = tile_exponential_sum;
                    }
                    __syncthreads();

                    // This thread owns one head dimension of the output.
                    float tile_numerator = 0.0F;
                    for (std::size_t token_in_tile = 0; token_in_tile < tile_token_count; ++token_in_tile) {
                        const auto shared_offset = token_in_tile * head_size + head_dimension_index;
                        tile_numerator += scores_shared[token_in_tile] * v_shared[shared_offset];
                    }

                    const auto updated_maximum = cuda::std::max(running_maximum, *tile_maximum_shared);
                    const auto previous_rescale = cuda::std::exp(running_maximum - updated_maximum);
                    const auto tile_rescale = cuda::std::exp(*tile_maximum_shared - updated_maximum);

                    running_numerator = running_numerator * previous_rescale + tile_numerator * tile_rescale;
                    running_denominator = running_denominator * previous_rescale + *tile_exponential_sum_shared * tile_rescale;
                    running_maximum = updated_maximum;
                    __syncthreads();
                }

                args.output(query_head_index, head_dimension_index) = __float2bfloat16(running_numerator / running_denominator);
            }
        };
    } // namespace detail

    inline void flash_attention(cuda::stream_ref stream, const TensorView<const __nv_bfloat16> query, const TensorView<const __nv_bfloat16> key,
                                const TensorView<const __nv_bfloat16> value, const TensorView<__nv_bfloat16> output, const std::size_t token_pos) {
        const auto q_head_count = query.dim(0);
        const auto head_size = query.dim(1);
        const auto shared_memory = cuda::dynamic_shared_memory<float[]>(detail::shared_memory_element_count(head_size));
        const auto config = cuda::make_config(cuda::grid_dims(q_head_count), cuda::block_dims(head_size), shared_memory);

        const auto args = detail::FlashAttentionBf16::Args{
            .query = query,
            .key = key,
            .value = value,
            .output = output,
            .token_pos = token_pos,
        };

        cuda::launch(stream, config, detail::FlashAttentionBf16{}, args);
    }
} // namespace inference::gpu::kernels
