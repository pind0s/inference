#pragma once
#include "backend/cuda/kernels/util.cuh"
#include "backend/rope_cache.hpp"
#include "tensor/tensor_view.hpp"
#include <cuda/cmath>
#include <cuda/launch>
#include <cuda_bf16.h>

namespace inference::gpu::kernels {
    namespace detail {
        struct RopeBf16 {

            struct Args {
                TensorView<__nv_bfloat16> values;
                RopeCacheView cache;
                std::size_t token_pos;
            };

            template <typename Configuration>
            __device__ void operator()(const Configuration& config, Args args) {
                const auto global_index = cuda::gpu_thread.rank(cuda::grid, config);
                const auto head_count = args.values.dim(0);
                const auto head_size = args.values.dim(1);
                const auto half_size = head_size / 2;
                const auto pair_count = head_count * half_size;

                if (global_index >= pair_count) {
                    return;
                }

                const auto head = global_index / half_size;
                const auto index = global_index % half_size;

                const float sin = args.cache.sine(args.token_pos, index);
                const float cos = args.cache.cosine(args.token_pos, index);

                const float first = __bfloat162float(args.values(head, index));
                const float second = __bfloat162float(args.values(head, half_size + index));

                args.values(head, index) = __float2bfloat16(first * cos - second * sin);
                args.values(head, half_size + index) = __float2bfloat16(first * sin + second * cos);
            }
        };
    } // namespace detail

    inline void rope(cuda::stream_ref stream, const TensorView<__nv_bfloat16> values, const RopeCacheView cache, const std::size_t token_pos) {
        const auto head_count = values.dim(0);
        const auto head_size = values.dim(1);
        const auto block_count = cuda::ceil_div(head_count * head_size / 2, util::threads_per_block);
        const auto config = cuda::make_config(cuda::grid_dims(block_count), cuda::block_dims<util::threads_per_block>());

        auto args = detail::RopeBf16::Args{
            .values = values,
            .cache = cache,
            .token_pos = token_pos,
        };

        cuda::launch(stream, config, detail::RopeBf16{}, args);
    }
} // namespace inference::gpu::kernels
