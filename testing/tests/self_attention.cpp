#include "reference/self_attention.hpp"
#include "backend/backend.hpp"
#include "types/device.hpp"
#include "util/make_tensor.hpp"
#include "util/random.hpp"
#include "util/tensor_matchers.hpp"
#include "util/to_host.hpp"
#include <algorithm>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <vector>

constexpr std::size_t query_head_count = 32;
constexpr std::size_t kv_head_count = 32;
constexpr std::size_t position = 1024;
constexpr std::size_t head_dim = 128;

TEST_CASE("flash decode attention", "[backend][self_attention][flash decode]") {
    using namespace inference;

    auto& backend = Backend::get_backend<types::Device::CUDA>();
    auto& cpu_backend = Backend::get_backend<types::Device::CPU>();

    const auto query = test::random_bf16({ query_head_count, head_dim }, backend);
    const auto key = test::random_bf16({ position + 1, kv_head_count, head_dim }, backend);
    const auto value = test::random_bf16({ position + 1, kv_head_count, head_dim }, backend);
    auto output = test::random_bf16({ query_head_count, head_dim }, backend);

    auto expected = test::reference::self_attention(query, key, value, position);
    backend.self_attention(query, key, value, output, position);
    auto output_host = test::to_host(output);

    CHECK_THAT(output_host, test::WithinRel(expected, 0.01f));

}

TEST_CASE("flash decode attention", "[benchmark][self_attention][flash decode]") {
    using namespace inference;

    const auto device = GENERATE(types::Device::CPU, types::Device::CUDA);
    CAPTURE(types::device_to_string(device));
    auto& backend = Backend::select_backend(device);

    const auto query = test::random_bf16({ query_head_count, head_dim }, backend);
    const auto key = test::random_bf16({ position + 1, kv_head_count, head_dim }, backend);
    const auto value = test::random_bf16({ position + 1, kv_head_count, head_dim }, backend);
    auto output = Tensor::empty(query.shape(), types::DType::BF16, backend);

    BENCHMARK(types::device_to_string(device) + " flash decode") {
        backend.self_attention(query, key, value, output, position);
        if (device == types::Device::CUDA) {
            if (const auto error = cudaDeviceSynchronize(); error != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(error));
            }
        }
    };
}
