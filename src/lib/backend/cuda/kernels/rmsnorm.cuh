#pragma once
#include "tensor/tensor_view.hpp"
#include <cub/block/block_reduce.cuh>
#include <cuda/cmath>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace inference::gpu::kernels {
    namespace detail {

        struct RmsNormKernel {
            using BlockReduce = cub::BlockReduce<float, util::threads_per_block>;

            template <typename Configuration>
            __device__ void operator()(const Configuration& config, TensorView<const __nv_bfloat16> input, TensorView<const __nv_bfloat16> weight,
                                       TensorView<__nv_bfloat16> output, float epsilon) {

                auto smem = cuda::dynamic_shared_memory(config);

                const auto row_size = weight.size();
                const auto row = cuda::block.rank(cuda::grid, config);
                const auto thread_index = cuda::gpu_thread.rank(cuda::block, config);
                const auto block_thread_count = cuda::gpu_thread.count(cuda::block, config);
                const auto row_base = row * row_size;

                float local_sum = 0.0f;
                for (std::size_t col = thread_index; col < row_size; col += block_thread_count) {
                    const float value = __bfloat162float(input[row_base + col]);
                    smem[col] = value;
                    local_sum += value * value;
                }

                __shared__ BlockReduce::TempStorage reduce_storage;
                __shared__ float scale;
                const float sum_of_squares = BlockReduce(reduce_storage).Sum(local_sum);

                if (thread_index == 0) {
                    scale = 1.0f / cuda::std::sqrtf(sum_of_squares / static_cast<float>(row_size) + epsilon);
                }
                __syncthreads();

                for (std::size_t col = thread_index; col < row_size; col += block_thread_count) {
                    auto weight_value = __bfloat162float(weight(col));
                    output[row_base + col] = __float2bfloat16(smem[col] * scale * weight_value);
                }
            }
        };
    } // namespace detail

    inline void rmsnorm(const cuda::stream_ref stream, const TensorView<const __nv_bfloat16> input, const TensorView<const __nv_bfloat16> weight,
                        const TensorView<__nv_bfloat16> output, float epsilon) {
        const auto row_size = weight.size();
        const auto row_count = input.size() / row_size;
        const auto shared_memory = cuda::dynamic_shared_memory<float[]>(row_size);
        const auto config = cuda::make_config(cuda::grid_dims(row_count), cuda::block_dims<util::threads_per_block>(), shared_memory);
        cuda::launch(stream, config, detail::RmsNormKernel{}, input, weight, output, epsilon);
    }
} // namespace inference::gpu::kernels
