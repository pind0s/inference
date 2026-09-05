#include "reference/self_attention.hpp"
#include "backend/backend.hpp"
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

TEST_CASE("self attention matches the reference", "[backend][self_attention]") {
    using namespace inference;

    const auto device = GENERATE(types::Device::CPU, types::Device::CUDA);
    CAPTURE(types::device_to_string(device));
    auto& backend = Backend::select_backend(device);

    const auto head_size = GENERATE(32uz, 65uz, 128uz);
    const auto kv_head_count = GENERATE(1uz, 2uz, 4uz);
    const auto position = GENERATE(0uz, 1uz, 31uz, 32uz, 50uz);
    CAPTURE(head_size, kv_head_count, position);

    constexpr std::size_t query_head_count = 4;
    constexpr std::size_t max_pos = 51;
    const auto query = test::random_bf16({ query_head_count, head_size }, backend);
    const auto key = test::random_bf16({ max_pos, kv_head_count, head_size }, backend, { 1 });
    const auto value = test::random_bf16({ max_pos, kv_head_count, head_size }, backend, { 2 });
    auto output = Tensor::empty(query.shape(), types::DType::BF16, backend);
    const auto expected = test::reference::self_attention(query, key, value, position);

    backend.self_attention(query, key, value, output, position);
    const auto output_host = test::to_host(output);
    CHECK_THAT(output_host, test::WithinRel(expected, 0.01f));
}

TEST_CASE("self attention benchmark", "[benchmark][self_attention]") {
    using namespace inference;

    const auto device = GENERATE(types::Device::CPU, types::Device::CUDA);
    CAPTURE(types::device_to_string(device));
    auto& backend = Backend::select_backend(device);

    constexpr std::size_t query_head_count = 8;
    constexpr std::size_t kv_head_count = 2;
    constexpr std::size_t head_size = 128;
    constexpr std::size_t position = 50;
    const auto query = test::random_bf16({ query_head_count, head_size }, backend);
    const auto key = test::random_bf16({ position + 1, kv_head_count, head_size }, backend);
    const auto value = test::random_bf16({ position + 1, kv_head_count, head_size }, backend);
    auto output = Tensor::empty(query.shape(), types::DType::BF16, backend);

    BENCHMARK(types::device_to_string(device) + " Self attention") {
        backend.self_attention(query, key, value, output, position);
        if (device == types::Device::CUDA) {
            if (const auto error = cudaDeviceSynchronize(); error != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(error));
            }
        }
    };
}
