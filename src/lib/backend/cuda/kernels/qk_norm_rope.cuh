#pragma once
#include "backend/util/rope_cache.hpp"
#include "tensor/tensor_view.hpp"
#include <cub/block/block_reduce.cuh>
#include <cuda/cmath>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace inference::gpu::kernels {
    namespace detail {

        struct QkNormRopeBf16 {
            struct QkSum {
                float query = 0.0f;
                float key = 0.0f;
            };

            using BlockReduce = cub::BlockReduce<QkSum, 128>;

            template <typename Configuration>
            __device__ void operator()(const Configuration& config, TensorView<__nv_bfloat16> query, TensorView<__nv_bfloat16> key,
                                       TensorView<const __nv_bfloat16> q_norm, TensorView<const __nv_bfloat16> k_norm,
                                       const backend::RopeCacheView& sin_cos, std::size_t token_pos, float eps) {

                const auto head_size = query.dim(1);
                const auto half_size = head_size / 2;
                const auto key_head_count = key.dim(0);
                const auto query_head_count = query.dim(0);
                const auto head = cuda::block.rank(cuda::grid, config);
                const auto index = cuda::gpu_thread.rank(cuda::block, config);

                QkSum local_sum;

                float query_value = 0.0f;
                if (head < query_head_count) {
                    query_value = __bfloat162float(query(head, index));
                    local_sum.query += query_value * query_value;
                }

                float key_value = 0.0f;
                if (head < key_head_count) {
                    key_value = __bfloat162float(key(head, index));
                    local_sum.key += key_value * key_value;
                }

                __shared__ BlockReduce::TempStorage reduce_storage;
                const auto sums = BlockReduce(reduce_storage).Reduce(local_sum, [](const QkSum& left, const QkSum& right) {
                    return QkSum{
                        .query = left.query + right.query,
                        .key = left.key + right.key,
                    };
                });

                __shared__ float scale_q;
                __shared__ float scale_k;
                if (index == 0) {
                    auto q_mean = sums.query / static_cast<float>(head_size);
                    auto k_mean = sums.key / static_cast<float>(head_size);
                    scale_q = 1.0F / cuda::std::sqrtf(q_mean + eps);
                    scale_k = 1.0F / cuda::std::sqrtf(k_mean + eps);
                }
                __syncthreads();

                if (index >= half_size) {
                    return;
                }

                const auto cos = sin_cos.cosine(token_pos, index);
                const auto sin = sin_cos.sine(token_pos, index);

                if (head < query_head_count) {

                    const float x = query_value * __bfloat162float(q_norm(index)) * scale_q;
                    const float y = __bfloat162float(query(head, half_size + index)) * __bfloat162float(q_norm(half_size + index)) * scale_q;

                    query(head, index) = __float2bfloat16(x * cos - y * sin);
                    query(head, half_size + index) = __float2bfloat16(y * cos + x * sin);
                }

                if (head < key_head_count) {
                    const float x = key_value * __bfloat162float(k_norm(index)) * scale_k;
                    const float y = __bfloat162float(key(head, half_size + index)) * __bfloat162float(k_norm(half_size + index)) * scale_k;

                    key(head, index) = __float2bfloat16(x * cos - y * sin);
                    key(head, half_size + index) = __float2bfloat16(y * cos + x * sin);
                }
            }
        };
    } // namespace detail

    inline void qk_norm_rope(const cuda::stream_ref stream, TensorView<__nv_bfloat16> query, TensorView<__nv_bfloat16> key,
                             TensorView<const __nv_bfloat16> q_norm, TensorView<const __nv_bfloat16> k_norm, const backend::RopeCacheView& sin_cos,
                             std::size_t token_pos, float eps) {
        // each block will handle 1 full row?
        const auto max_head_count = std::max(query.dim(0), key.dim(0));
        const auto config = cuda::make_config(cuda::grid_dims(max_head_count), cuda::block_dims<128>());
        cuda::launch(stream, config, detail::QkNormRopeBf16{}, query, key, q_norm, k_norm, sin_cos, token_pos, eps);
    }
} // namespace inference::gpu::kernels
