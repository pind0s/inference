#include <benchmark/benchmark.h>
#include <common/fake_qwen3.hpp>

namespace test {
    static void qwen3_decode(benchmark::State& state) {
        using namespace inference;

        auto& backend = Backend::get_backend<types::Device::CPU>();
        const auto context_length = static_cast<std::size_t>(state.range(0));
        auto [model, kv_cache] = util::make_fake_qwen3(backend, context_length + 1);

        types::TokenId token = 1;
        for (std::size_t position = 0; position < context_length; ++position) {
            token = model.forward(token, backend, kv_cache);
        }

        for (auto _ : state) {
            kv_cache.token_count = context_length;
            auto next_token = model.forward(token, backend, kv_cache);
            benchmark::DoNotOptimize(next_token);
            benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(state.iterations());
    }

    BENCHMARK(qwen3_decode)->Arg(1)->Arg(128)->Arg(512)->Arg(2048)->ArgName("context")->Unit(benchmark::kMillisecond);
} // namespace test
