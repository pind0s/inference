#include "backend/cpu/kernels/elementwise.hpp"
#include <benchmark/benchmark.h>
#include <common/random.hpp>
#include <vector>

namespace test {
    static void add(benchmark::State& state) {
        using namespace inference;

        const auto element_count = static_cast<std::size_t>(state.range(0));
        const TensorShape shape(element_count);
        const auto lhs = util::random_bf16_tensor(shape);
        const auto rhs = util::random_bf16_tensor(shape);
        std::vector<cpu::bf16_t> output(element_count);

        for (auto _ : state) {
            cpu::kernels::add(lhs.view<cpu::bf16_t>(), rhs.view<cpu::bf16_t>(), TensorView(output.data(), shape));
            benchmark::DoNotOptimize(output.data());
            benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(element_count));
    }

    BENCHMARK(add)->RangeMultiplier(4)->Range(16, 1 << 20);
} // namespace test
