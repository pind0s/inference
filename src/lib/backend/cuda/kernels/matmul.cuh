#pragma once
#include "backend/cuda/kernels/util.cuh"
#include "tensor/tensor_view.hpp"
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cuda/barrier>
#include <cuda/cmath>
#include <cuda/launch>
#include <cuda/stream>
#include <cuda_bf16.h>
#include <mma.h>

namespace inference::gpu::kernels {
    namespace detail {
        // todo switch from cublas to my own matmul impl
        // struct MatmulBf16 {
        //     struct Args {
        //         TensorView<const __nv_bfloat16> input;
        //         TensorView<const __nv_bfloat16> weights;
        //         TensorView<__nv_bfloat16> output;
        //     };
        //
        //     static constexpr std::size_t WMMA_M = 16;
        //     static constexpr std::size_t WMMA_N = 16;
        //     static constexpr std::size_t WMMA_K = 16;
        //     static constexpr std::size_t WARP_SIZE = 32;
        //     static constexpr std::size_t WARPS_PER_BLOCK = util::threads_per_block / WARP_SIZE;
        //     static constexpr std::size_t TILE_M = WMMA_M;
        //     static constexpr std::size_t TILE_N = WARPS_PER_BLOCK * WMMA_N;
        //     static constexpr std::size_t TILE_K = WMMA_K;
        //
        //     template <typename Configuration>
        //     __device__ void operator()(const Configuration& config, const Args args) const noexcept {
        //         using namespace nvcuda;
        //
        //         // auto block = cooperative_groups::this_thread_block();
        //         const auto global_thread_index = cuda::gpu_thread.rank(cuda::grid, config);
        //         const auto block_thread_index = cuda::gpu_thread.rank(cuda::block, config);
        //         const auto block_index = cuda::block.rank(cuda::grid, config);
        //
        //         const auto warp_id = block_thread_index / WARP_SIZE; // warp index within block
        //         const auto lane_id = block_thread_index % WARP_SIZE; // thread index within warp
        //
        //         const auto* input_ptr = args.input.data();
        //         const auto* weights_ptr = args.weights.data();
        //         auto* output_ptr = args.output.data();
        //
        //         const auto K = args.weights.dim(1);
        //         const auto N = args.weights.dim(0);
        //         const auto M = args.input.size() / K;
        //
        //         const auto num_n_tiles = cuda::ceil_div(N, TILE_N);
        //         const auto m_block_id = block_index / num_n_tiles;
        //         const auto n_block_id = block_index % num_n_tiles;
        //
        //         const auto m_start = m_block_id * TILE_M;
        //         const auto n_start = n_block_id * TILE_N + warp_id * WMMA_N;
        //
        //         if (m_start >= M || n_start >= N) {
        //             return;
        //         }
        //
        //         __shared__ float output_tiles[WARPS_PER_BLOCK][WMMA_M * WMMA_N];
        //
        //         wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> accumulator;
        //         wmma::fill_fragment(accumulator, 0.0f);
        //
        //         for (std::size_t k_start = 0; k_start < K; k_start += TILE_K) {
        //             wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> matrix_b;
        //             wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> matrix_a;
        //
        //             const auto* a_tile_start = input_ptr + m_start * K + k_start;
        //             const auto* b_tile_start = weights_ptr + n_start * K + k_start;
        //             wmma::load_matrix_sync(matrix_a, a_tile_start, K);
        //             wmma::load_matrix_sync(matrix_b, b_tile_start, K);
        //             wmma::mma_sync(accumulator, matrix_a, matrix_b, accumulator);
        //         }
        //
        //         wmma::store_matrix_sync(output_tiles[warp_id], accumulator, WMMA_N, wmma::mem_row_major);
        //         __syncwarp();
        //
        //         for (std::size_t index = lane_id; index < WMMA_M * WMMA_N; index += WARP_SIZE) {
        //             const auto local_row = index / WMMA_N;
        //             const auto local_column = index % WMMA_N;
        //             const auto global_row = m_start + local_row;
        //             const auto global_column = n_start + local_column;
        //
        //             if (global_row < M && global_column < N) {
        //                 output_ptr[global_row * N + global_column] = __float2bfloat16(output_tiles[warp_id][index]);
        //             }
        //         }
        //
        //         // if (global_index >= output_size) {
        //         //     return;
        //         // }
        //         //
        //         // const auto row = global_index / columns;
        //         // const auto column = global_index % columns;
        //         // float sum = 0.0F;
        //         //
        //         // for (std::size_t inner = 0; inner < inner_size; ++inner) {
        //         //     const auto input_value = __bfloat162float(args.input[row * inner_size + inner]);
        //         //     const auto weight_value = __bfloat162float(args.weights[column * inner_size + inner]);
        //         //     sum += input_value * weight_value;
        //         // }
        //         //
        //         // args.output[global_index] = __float2bfloat16(sum);
        //     }
        // };
    } // namespace detail

    // inline void matmul_bf16(const cuda::stream_ref stream, const TensorView<const __nv_bfloat16> input, const TensorView<const __nv_bfloat16> weights,
    //                         const TensorView<__nv_bfloat16> output) {
    //     const auto K = weights.dim(1);
    //     const auto N = weights.dim(0);
    //     const auto M = input.size() / K;
    //     const auto m_block_count = cuda::ceil_div(M, detail::MatmulBf16::TILE_M);
    //     const auto n_block_count = cuda::ceil_div(N, detail::MatmulBf16::TILE_N);
    //     const auto block_count = m_block_count * n_block_count;
    //     const auto config = cuda::make_config(cuda::grid_dims(block_count), cuda::block_dims<util::threads_per_block>());
    //     const auto args = detail::MatmulBf16::Args{
    //         .input = input,
    //         .weights = weights,
    //         .output = output,
    //     };
    //
    //     cuda::launch(stream, config, detail::MatmulBf16{}, args);
    // }

    inline void matmul_cublas_bf16(const cublasHandle_t handle, const TensorView<const __nv_bfloat16> input, const TensorView<const __nv_bfloat16> weights,
                                   const TensorView<__nv_bfloat16> output) {
        const auto inner_size = weights.dim(1);
        const auto columns = weights.dim(0);
        const auto rows = input.size() / inner_size;
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(columns), static_cast<int>(rows), static_cast<int>(inner_size), &alpha,
                     weights.data(), CUDA_R_16BF, static_cast<int>(inner_size), input.data(), CUDA_R_16BF, static_cast<int>(inner_size), &beta,
                     output.data(), CUDA_R_16BF, static_cast<int>(columns), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
} // namespace inference::gpu::kernels
