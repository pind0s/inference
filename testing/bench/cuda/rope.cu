#include "backend/cuda/kernels/rope.cuh"
#include "backend/rope_cache.hpp"
#include "common/random.hpp"
#include "cuda_event.cuh"
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cuda/devices>
#include <cuda/stream>
#include <cuda_bf16.h>

namespace test {
    namespace {
        constexpr auto theta = 1'000'000.0F;
        constexpr std::size_t position = 1337;

        void cuda_rope(benchmark::State& state) {
            using namespace inference;
            auto& backend = Backend::get_backend<types::Device::CUDA>();
            const cuda::stream stream(cuda::devices[0]);
            const auto head_count = static_cast<std::size_t>(state.range(0));
            const auto head_size = static_cast<std::size_t>(state.range(1));

            const auto rope_cache = make_rope_cache(position + 1, head_size, theta, backend);
            auto values = util::random_bf16_tensor({ head_count, head_size }).host_to_device();

            gpu::kernels::rope(stream, values.view<__nv_bfloat16>(), rope_cache.view(), position);
            stream.sync();

            for (auto _ : state) {
                const CudaEventPair events;
                events.record_start(stream);
                gpu::kernels::rope(stream, values.view<__nv_bfloat16>(), rope_cache.view(), position);
                const auto elapsed_seconds = events.record_stop_and_elapsed_seconds(stream);
                state.SetIterationTime(elapsed_seconds);
            }

            state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(values.size()));
        }
    } // namespace

    BENCHMARK(cuda_rope)->Args({ 8, 128 })->Args({ 16, 128 })->Args({ 32, 128 })->ArgNames({ "heads", "head_size" })->UseManualTime();
} // namespace test
