#pragma once
#include "tensor/tensor_view.hpp"
#include "types/token.hpp"
#include <cuda/launch>
#include <cuda_bf16.h>

namespace inference::gpu::kernels {
    namespace detail {
        struct EmbeddingBf16 {
            template <typename Configuration>
            __device__ void operator()(const Configuration& config, const TensorView<const __nv_bfloat16> weights, const TensorView<__nv_bfloat16> output,
                                       const types::TokenId token_id) const noexcept {
                const auto thread_index = cuda::gpu_thread.rank(cuda::grid, config);
                const auto thread_count = cuda::gpu_thread.count(cuda::grid, config);
                const auto hidden_size = output.size();

                for (std::size_t hidden = thread_index; hidden < hidden_size; hidden += thread_count) {
                    output(hidden) = weights(token_id, hidden);
                }
            }
        };
    } // namespace detail

    // todo this should just be a cuda::copy_bytes or something?
    inline void embedding(const cuda::stream_ref stream, const TensorView<const __nv_bfloat16> weights, const TensorView<__nv_bfloat16> output,
                          const types::TokenId token_id) {
        const auto hidden_size = output.size();
        const auto block_count = cuda::ceil_div(hidden_size, util::threads_per_block);
        const auto config = cuda::make_config(cuda::grid_dims(block_count), cuda::block_dims<util::threads_per_block>());
        cuda::launch(stream, config, detail::EmbeddingBf16{}, weights, output, token_id);
    }
} // namespace inference::gpu::kernels
