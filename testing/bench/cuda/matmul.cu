// #include "backend/cuda/kernels/matmul.cuh"
// #include "common/random.hpp"
// #include "cuda_event.cuh"
// #include <benchmark/benchmark.h>
// #include <cstddef>
// #include <cstdint>
// #include <cublas_v2.h>
// #include <cuda/devices>
// #include <cuda/stream>
// #include <cuda_bf16.h>
//
// namespace test {
//     void cublas_matmul_bf16(const cublasHandle_t& handle, const inference::TensorView<const __nv_bfloat16> input,
//                             const inference::TensorView<const __nv_bfloat16> weights, const inference::TensorView<__nv_bfloat16> output) {
//         const auto inner_size = weights.dim(1);
//         const auto columns = weights.dim(0);
//         const auto rows = input.size() / inner_size;
//         constexpr float alpha = 1.0F;
//         constexpr float beta = 0.0F;
//         cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(columns), static_cast<int>(rows), static_cast<int>(inner_size), &alpha,
//                      weights.data(), CUDA_R_16BF, static_cast<int>(inner_size), input.data(), CUDA_R_16BF, static_cast<int>(inner_size), &beta,
//                      output.data(), CUDA_R_16BF, static_cast<int>(columns), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
//     }
//
//     template <typename Matmul>
//     void run_matmul_benchmark(benchmark::State& state, const cuda::stream_ref stream, const inference::TensorView<const __nv_bfloat16> input,
//                               const inference::TensorView<const __nv_bfloat16> weights, const inference::TensorView<__nv_bfloat16> output,
//                               Matmul&& matmul) {
//
//         matmul(input, weights, output);
//         stream.sync();
//
//         for (auto _ : state) {
//             const CudaEventPair events;
//
//             events.record_start(stream);
//             matmul(input, weights, output);
//             const auto elapsed_seconds = events.record_stop_and_elapsed_seconds(stream);
//             state.SetIterationTime(elapsed_seconds);
//         }
//
//         const auto rows = input.size() / weights.dim(1);
//         const auto operation_count = 2 * rows * weights.dim(0) * weights.dim(1);
//         state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(operation_count));
//     }
//
//     void cuda_matmul(benchmark::State& state) {
//         using namespace inference;
//
//         const auto rows = static_cast<std::size_t>(state.range(0));
//         const auto columns = static_cast<std::size_t>(state.range(1));
//         const auto inner_size = static_cast<std::size_t>(state.range(2));
//         const cuda::stream stream(cuda::devices[0]);
//         const auto input = util::cpu::random_bf16_tensor({ rows, inner_size }).host_to_device();
//         const auto weights = util::cpu::random_bf16_tensor({ columns, inner_size }).host_to_device();
//         auto& backend = Backend::get_backend<types::Device::CUDA>();
//         auto output = Tensor::empty({ rows, columns }, types::DType::BF16, backend);
//
//         run_matmul_benchmark(state, stream, input.view<__nv_bfloat16>(), weights.view<__nv_bfloat16>(), output.view<__nv_bfloat16>(),
//                              [&](const auto input_view, const auto weights_view, const auto output_view) {
//                                  gpu::kernels::matmul_bf16(stream, input_view, weights_view, output_view);
//                              });
//     }
//
//     static void cublas_matmul(benchmark::State& state) {
//         using namespace inference;
//
//         const auto rows = static_cast<std::size_t>(state.range(0));
//         const auto columns = static_cast<std::size_t>(state.range(1));
//         const auto inner_size = static_cast<std::size_t>(state.range(2));
//
//         const cuda::stream stream(cuda::devices[0]);
//         const auto input = util::cpu::random_bf16_tensor({ rows, inner_size }).host_to_device();
//         const auto weights = util::cpu::random_bf16_tensor({ columns, inner_size }).host_to_device();
//         auto& backend = Backend::get_backend<types::Device::CUDA>();
//         auto output = Tensor::empty({ rows, columns }, types::DType::BF16, backend);
//         cublasHandle_t handle{};
//         cublasCreate(&handle);
//         cublasSetStream(handle, stream.get());
//
//         run_matmul_benchmark(state, stream, input.view<__nv_bfloat16>(), weights.view<__nv_bfloat16>(), output.view<__nv_bfloat16>(),
//                              [&](const auto input_view, const auto weights_view, const auto output_view) {
//                                  cublas_matmul_bf16(handle, input_view, weights_view, output_view);
//                              });
//
//         cublasDestroy(handle);
//     }
//
//     BENCHMARK(test::cuda_matmul)
//         ->Args({ 1, 1024, 1024 })
//         ->Args({ 16, 1024, 1024 })
//         ->Args({ 128, 1024, 1024 })
//         ->ArgNames({ "m", "n", "k" })
//         ->UseManualTime();
//
//     BENCHMARK(test::cublas_matmul)
//         ->Args({ 1, 1024, 1024 })
//         ->Args({ 16, 1024, 1024 })
//         ->Args({ 128, 1024, 1024 })
//         ->ArgNames({ "m", "n", "k" })
//         ->UseManualTime();
// } // namespace test
