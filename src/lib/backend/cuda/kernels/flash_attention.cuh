#pragma once
#include "tensor/tensor.hpp"
#include "tensor/tensor_view.hpp"
#include <cassert>
#include <cub/block/block_reduce.cuh>
#include <cuda/cmath>
#include <cuda/launch>
#include <cuda/std/algorithm>
#include <cuda_bf16.h>

namespace inference::gpu::kernels {
    inline constexpr std::size_t KV_TILE_SIZE = 16;

    struct FlashAttentionScratchView {
        TensorView<float> maximum;
        TensorView<float> denominator;
        TensorView<float> numerator;
    };

    struct FlashAttentionScratch {
        Tensor maximum;
        Tensor denominator;
        Tensor numerator;

        [[nodiscard]] FlashAttentionScratchView view() noexcept {
            return {
                .maximum = maximum.view<float>(),
                .denominator = denominator.view<float>(),
                .numerator = numerator.view<float>(),
            };
        }
    };

    namespace detail {
        constexpr int threads_per_block = 128;
        using BlockReduce = cub::BlockReduce<float, threads_per_block>;

        struct FlashAttentionDecodeBF16 {
            struct Args {
                TensorView<const __nv_bfloat16> query;
                TensorView<const __nv_bfloat16> key;
                TensorView<const __nv_bfloat16> value;
                FlashAttentionScratchView workspace;
                std::size_t token_pos;
            };

            template <typename Configuration>
            __device__ void operator()(const Configuration& config, Args args) {
                const auto context_len = args.token_pos + 1;
                const auto tile_per_head = cuda::ceil_div(context_len, KV_TILE_SIZE);

                const auto block_index = cuda::block.rank(cuda::grid, config);
                const auto thread_index = cuda::gpu_thread.rank(cuda::block, config);

                const auto q_head_index = block_index / tile_per_head;
                const auto q_head_count = args.query.dim(0);
                const auto kv_head_count = args.key.dim(1);
                const auto kv_head_index = q_head_index / (q_head_count / kv_head_count);
                const auto tile_index = block_index % tile_per_head;
                const auto first_token = tile_index * KV_TILE_SIZE;

                auto tile_token_count = KV_TILE_SIZE;
                tile_token_count = cuda::std::min(tile_token_count, context_len - first_token);

                constexpr int head_dim = 128;
                const auto attention_scale = 1.0f / cuda::std::sqrtf(static_cast<float>(head_dim));
                float q_value = __bfloat162float(args.query(q_head_index, thread_index));

                __shared__ BlockReduce::TempStorage reduce_storage;
                __shared__ float scores[KV_TILE_SIZE];
                __shared__ float tile_max;

                float local_max = -cuda::std::numeric_limits<float>::infinity();
                for (std::size_t tile = 0; tile < tile_token_count; ++tile) {
                    auto token_idx = first_token + tile;
                    float dot = q_value * __bfloat162float(args.key(token_idx, kv_head_index, thread_index));

                    const float score = BlockReduce(reduce_storage).Sum(dot);
                    if (thread_index == 0) {
                        scores[tile] = score * attention_scale;
                        local_max = cuda::std::max(local_max, scores[tile]);
                    }
                }

                if (thread_index == 0) {
                    tile_max = local_max;
                }
                __syncthreads();

                float local_denominator = 0.0f;
                if (thread_index < tile_token_count) {
                    const float attention = cuda::std::expf(scores[thread_index] - tile_max);
                    scores[thread_index] = attention;
                    local_denominator = attention;
                }
                const float denominator = BlockReduce(reduce_storage).Sum(local_denominator);

                if (thread_index == 0) {
                    args.workspace.maximum(q_head_index, tile_index) = tile_max;
                    args.workspace.denominator(q_head_index, tile_index) = denominator;
                }

                float numerator = 0.0f;
                for (std::size_t tile = 0; tile < tile_token_count; ++tile) {
                    numerator += scores[tile] * __bfloat162float(args.value(first_token + tile, kv_head_index, thread_index));
                }
                args.workspace.numerator(q_head_index, tile_index, thread_index) = numerator;
            }
        };

        struct CombineFlashAttention {
            struct Args {
                FlashAttentionScratchView workspace;
                TensorView<__nv_bfloat16> output;
                std::size_t active_splits;
            };

            template <typename Configuration>
            __device__ void operator()(const Configuration& config, Args args) const {
                const auto head_index = cuda::block.rank(cuda::grid, config);
                const auto thread_index = cuda::gpu_thread.rank(cuda::block, config);
                const auto num_splits = args.active_splits;

                __shared__ float maximum;
                if (thread_index == 0) {
                    float local_maximum = -cuda::std::numeric_limits<float>::infinity();
                    for (std::size_t index = 0; index < num_splits; ++index) {
                        local_maximum = cuda::std::max(local_maximum, args.workspace.maximum(head_index, index));
                    }
                    maximum = local_maximum;
                }
                __syncthreads();

                float local_denominator = 0.0f;
                for (std::size_t split = thread_index; split < num_splits; split += threads_per_block) {
                    const float rescale = cuda::std::expf(args.workspace.maximum(head_index, split) - maximum);
                    local_denominator += args.workspace.denominator(head_index, split) * rescale;
                }

                __shared__ BlockReduce::TempStorage reduce_storage;
                __shared__ float denominator;
                const float reduce = BlockReduce(reduce_storage).Sum(local_denominator);

                if (thread_index == 0) {
                    denominator = reduce;
                }
                __syncthreads();

                float numerator = 0.0f;
                for (std::size_t split = 0; split < num_splits; ++split) {
                    const float rescale = cuda::std::expf(args.workspace.maximum(head_index, split) - maximum);
                    numerator += rescale * args.workspace.numerator(head_index, split, thread_index);
                }
                args.output(head_index, thread_index) = __float2bfloat16(numerator / denominator);
            }
        };
    } // namespace detail

    inline void new_flash_attention(cuda::stream_ref stream, const TensorView<const __nv_bfloat16> query, const TensorView<const __nv_bfloat16> key,
                                    const TensorView<const __nv_bfloat16> value, const TensorView<__nv_bfloat16> output,
                                    const FlashAttentionScratchView workspace, const std::size_t token_pos) {
        assert(query.dim(1) == 128);
        const auto q_num_heads = query.dim(0);
        const auto kv_splits = cuda::ceil_div(token_pos + 1, KV_TILE_SIZE);

        const auto num_blocks = q_num_heads * kv_splits;
        const auto split_config = cuda::make_config(cuda::grid_dims(num_blocks), cuda::block_dims(detail::threads_per_block));
        const auto split_args = detail::FlashAttentionDecodeBF16::Args{
            .query = query,
            .key = key,
            .value = value,
            .workspace = workspace,
            .token_pos = token_pos,
        };
        cuda::launch(stream, split_config, detail::FlashAttentionDecodeBF16{}, split_args);

        const auto combine_config = cuda::make_config(cuda::grid_dims(q_num_heads), cuda::block_dims(detail::threads_per_block));
        const auto combine_args = detail::CombineFlashAttention::Args{
            .workspace = workspace,
            .output = output,
            .active_splits = kv_splits,
        };
        cuda::launch(stream, combine_config, detail::CombineFlashAttention{}, combine_args);
    }
} // namespace inference::gpu::kernels
