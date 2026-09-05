#include "reference/rope.hpp"
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

constexpr std::size_t head_size = 128;
constexpr float theta = 1'000'000.0f;

TEST_CASE("rope matches the reference", "[backend][rope]") {
    using namespace inference;

    const auto device = GENERATE(types::Device::CPU, types::Device::CUDA);
    CAPTURE(types::device_to_string(device));
    auto& backend = Backend::select_backend(device);

    const auto position = GENERATE(0uz, 1uz, 50uz);
    CAPTURE(position);

    constexpr std::size_t head_count = 5;
    constexpr std::size_t max_pos = 51;
    auto input = test::random_bf16({ head_count, head_size }, backend);
    const auto cache = backend::make_rope_cache(max_pos, head_size, theta, backend);
    const auto expected = test::reference::rope(input, cache, position);

    backend.rope(input, cache, position);
    const auto output_host = test::to_host(input);
    CHECK_THAT(output_host, test::WithinRel(expected, 0.01f));
}

TEST_CASE("rope benchmark", "[benchmark][rope]") {
    using namespace inference;

    const auto device = GENERATE(types::Device::CPU, types::Device::CUDA);
    CAPTURE(types::device_to_string(device));
    auto& backend = Backend::select_backend(device);

    constexpr std::size_t head_count = 8;
    constexpr std::size_t position = 50;
    auto input = test::random_bf16({ head_count, head_size }, backend);
    const auto cache = backend::make_rope_cache(position + 1, head_size, theta, backend);

    BENCHMARK(types::device_to_string(device) + " RoPE") {
        backend.rope(input, cache, position);
        if (device == types::Device::CUDA) {
            if (const auto error = cudaDeviceSynchronize(); error != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(error));
            }
        }
    };
}
