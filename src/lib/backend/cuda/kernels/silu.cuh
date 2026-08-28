#pragma once
#include "tensor/tensor_view.hpp"
#include "util.cuh"
#include <cuda/cmath>
#include <cuda/launch>
#include <cuda_bf16.h>

namespace inference::gpu::kernels {
    namespace detail {
        struct SiluMultiplyBf16 {
            template <typename Configuration>
            __device__ void operator()(const Configuration& config, TensorView<const __nv_bfloat16> gate, TensorView<const __nv_bfloat16> up,
                                       TensorView<__nv_bfloat16> output) {
                const auto thread_index = cuda::gpu_thread.rank(cuda::grid, config);
                const auto thread_count = cuda::gpu_thread.count(cuda::grid, config);
                const auto element_count = output.size();

                for (std::size_t index = thread_index; index < element_count; index += thread_count) {
                    const auto gate_value = __bfloat162float(gate[index]);
                    const auto silu = gate_value / (1.0F + cuda::std::exp(-gate_value));
                    output[index] = __float2bfloat16(silu * __bfloat162float(up[index]));
                }
            }
        };
    } // namespace detail

    inline void silu(const cuda::stream_ref stream, TensorView<const __nv_bfloat16> gate, TensorView<const __nv_bfloat16> up,
                     TensorView<__nv_bfloat16> output) {
        const auto element_count = output.size();
        const auto block_count = cuda::ceil_div(element_count, util::threads_per_block);
        const auto config = cuda::make_config(cuda::grid_dims(block_count), cuda::block_dims<util::threads_per_block>());
        cuda::launch(stream, config, detail::SiluMultiplyBf16{}, gate, up, output);
    }
} // namespace inference::gpu::kernels
