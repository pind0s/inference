#include "backend/cpu/kernels/rope.hpp"
#include "backend/rope_cache.hpp"
#include "common/random.hpp"
#include <benchmark/benchmark.h>

namespace test {
    constexpr auto theta = 1'000'000.0F;
    constexpr auto position = 1337uz;

    static void rope(benchmark::State& state) {
        using namespace inference;

        const auto head_count = static_cast<std::size_t>(state.range(0));
        const auto head_size = static_cast<std::size_t>(state.range(1));
        const TensorShape shape(head_count, head_size);
        auto values = util::random_bf16_tensor(shape);
        auto& backend = Backend::get_backend<types::Device::CPU>();
        auto rope_cache = make_rope_cache(position + 1, head_size, theta, backend);

        for (auto _ : state) {
            cpu::kernels::rope(values.view<cpu::bf16_t>(), rope_cache.view(), position);
            benchmark::DoNotOptimize(values.data<cpu::bf16_t>());
            benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(values.size()));
    }

    BENCHMARK(rope)->Args({ 8, 128 })->Args({ 16, 128 })->ArgNames({ "heads", "head_size" });
} // namespace test
