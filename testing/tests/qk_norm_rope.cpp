#include "reference/qk_norm_rope.hpp"
#include "backend/backend.hpp"
#include "backend/util/rope_cache.hpp"
#include "types/device.hpp"
#include "util/random.hpp"
#include "util/tensor_matchers.hpp"
#include "util/to_host.hpp"
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <stdexcept>

constexpr std::size_t query_head_count = 8;
constexpr std::size_t head_size = 128;
constexpr float theta = 1'000'000.0f;
constexpr float epsilon = 1e-6f;

TEST_CASE("qk norm rope matches the reference", "[backend][qk_norm_rope]") {
    using namespace inference;

    const auto device = GENERATE(types::Device::CPU, types::Device::CUDA);
    CAPTURE(types::device_to_string(device));
    auto& backend = Backend::select_backend(device);

    const auto key_head_count = GENERATE(2uz, 8uz);
    const auto position = GENERATE(0uz, 1uz, 50uz);
    CAPTURE(key_head_count, position);

    auto query = test::random_bf16({ query_head_count, head_size }, backend);
    auto key = test::random_bf16({ key_head_count, head_size }, backend, { 1 });
    const auto q_norm = test::random_bf16({ head_size }, backend, { 2 });
    const auto k_norm = test::random_bf16({ head_size }, backend, { 3 });
    const auto cache = backend::make_rope_cache(position + 1, head_size, theta, backend);
    const auto expected_query = test::reference::norm_rope(query, q_norm, cache, position, epsilon);
    const auto expected_key = test::reference::norm_rope(key, k_norm, cache, position, epsilon);

    backend.qk_norm_rope(query, key, q_norm, k_norm, cache, position, epsilon);
    const auto query_host = test::to_host(query);
    const auto key_host = test::to_host(key);
    // Separate RMSNorm rounds to BF16 before RoPE, which matters near cancellation.
    CHECK_THAT(query_host, test::WithinRel(expected_query, 0.01f, 0.01f));
    CHECK_THAT(key_host, test::WithinRel(expected_key, 0.01f, 0.01f));
}

TEST_CASE("qk norm rope benchmark", "[benchmark][qk_norm_rope]") {
    using namespace inference;

    const auto device = GENERATE(types::Device::CPU, types::Device::CUDA);
    CAPTURE(types::device_to_string(device));
    auto& backend = Backend::select_backend(device);

    constexpr std::size_t key_head_count = 2;
    constexpr std::size_t position = 50;
    auto query = test::random_bf16({ query_head_count, head_size }, backend);
    auto key = test::random_bf16({ key_head_count, head_size }, backend);
    const auto q_norm = test::random_bf16({ head_size }, backend);
    const auto k_norm = test::random_bf16({ head_size }, backend);
    const auto cache = backend::make_rope_cache(position + 1, head_size, theta, backend);

    BENCHMARK(types::device_to_string(device) + " QK Norm RoPE") {
        backend.qk_norm_rope(query, key, q_norm, k_norm, cache, position, epsilon);
        if (device == types::Device::CUDA) {
            if (const auto error = cudaDeviceSynchronize(); error != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(error));
            }
        }
    };
}
