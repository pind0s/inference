#pragma once
#include "tensor/tensor_view.hpp"
#include <cub/block/block_reduce.cuh>
#include <cuda/cmath>
#include <cuda/launch>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace inference::gpu::kernels {
    namespace detail {

        struct RmsNormBf16 {
            using BlockReduce = cub::BlockReduce<float, 256>;

            template <typename Configuration>
            __device__ void operator()(const Configuration& config, TensorView<__nv_bfloat16> residual, TensorView<const __nv_bfloat16> weight,
                                       TensorView<__nv_bfloat16> output, float epsilon) {

                auto smem = cuda::dynamic_shared_memory(config);

                const auto hidden_size = residual.size();
                const auto thread_index = cuda::gpu_thread.rank(cuda::block, config);
                const auto block_thread_count = cuda::gpu_thread.count(cuda::block, config);

                float local_sum = 0.0f;
                for (std::size_t index = thread_index; index < hidden_size; index += block_thread_count) {
                    const float value = __bfloat162float(residual[index]);
                    smem[index] = value;
                    local_sum += value * value;
                }

                __shared__ BlockReduce::TempStorage reduce_storage;
                __shared__ float scale;
                const float sum_of_squares = BlockReduce(reduce_storage).Sum(local_sum);

                if (thread_index == 0) {
                    scale = 1.0f / cuda::std::sqrtf(sum_of_squares / static_cast<float>(hidden_size) + epsilon);
                }
                __syncthreads();

                for (std::size_t index = thread_index; index < hidden_size; index += block_thread_count) {
                    auto weight_value = __bfloat162float(weight(index));
                    output[index] = __float2bfloat16(smem[index] * scale * weight_value);
                }
            }
        };

        struct RmsNormAddBf16 {
            using BlockReduce = cub::BlockReduce<float, 256>;

            template <typename Configuration>
            __device__ void operator()(const Configuration& config, TensorView<__nv_bfloat16> residual, TensorView<const __nv_bfloat16> residual_update,
                                       TensorView<const __nv_bfloat16> weight, TensorView<__nv_bfloat16> output, float epsilon) {

                const auto hidden_size = residual.size();
                const auto thread_index = cuda::gpu_thread.rank(cuda::block, config);
                const auto block_thread_count = cuda::gpu_thread.count(cuda::block, config);

                float local_sum = 0.0f;
                for (std::size_t index = thread_index; index < hidden_size; index += block_thread_count) {
                    const float value = __bfloat162float(residual[index]) + __bfloat162float(residual_update[index]);
                    residual[index] = __float2bfloat16(value);
                    local_sum += value * value;
                }

                __shared__ BlockReduce::TempStorage reduce_storage;
                __shared__ float scale;
                const float sum_of_squares = BlockReduce(reduce_storage).Sum(local_sum);

                if (thread_index == 0) {
                    scale = 1.0f / cuda::std::sqrtf(sum_of_squares / static_cast<float>(hidden_size) + epsilon);
                }
                __syncthreads();

                for (std::size_t index = thread_index; index < hidden_size; index += block_thread_count) {
                    auto weight_value = __bfloat162float(weight(index));
                    output[index] = __float2bfloat16(__bfloat162float(residual[index]) * scale * weight_value);
                }
            }
        };
    } // namespace detail

    inline void rmsnorm(const cuda::stream_ref stream, const TensorView<__nv_bfloat16> input, const TensorView<const __nv_bfloat16> weight,
                        const TensorView<__nv_bfloat16> output, float epsilon) {
        const auto hidden_size = input.size();
        const auto shared_memory = cuda::dynamic_shared_memory<float[]>(hidden_size);
        const auto config = cuda::make_config(cuda::grid_dims(1), cuda::block_dims<256>(), shared_memory);
        cuda::launch(stream, config, detail::RmsNormBf16{}, input, weight, output, epsilon);
    }

    inline void rmsnorm_add(const cuda::stream_ref stream, const TensorView<__nv_bfloat16> residual, const TensorView<const __nv_bfloat16> residual_update,
                            const TensorView<const __nv_bfloat16> weight, const TensorView<__nv_bfloat16> output, float epsilon) {
        const auto config = cuda::make_config(cuda::grid_dims(1), cuda::block_dims<256>());
        cuda::launch(stream, config, detail::RmsNormAddBf16{}, residual, residual_update, weight, output, epsilon);
    }
} // namespace inference::gpu::kernels
