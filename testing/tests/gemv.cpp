#include "backend/backend.hpp"
#include "reference/matmul.hpp"
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

TEST_CASE("gemv matches the reference", "[backend][gemv]") {
    using namespace inference;

    const auto device = GENERATE(types::Device::CPU, types::Device::CUDA);
    CAPTURE(types::device_to_string(device));
    auto& backend = Backend::select_backend(device);

    const auto inner_size = GENERATE(32uz, 64uz, 128uz);
    const auto output_size = GENERATE(1uz, 7uz, 8uz, 9uz);
    CAPTURE(inner_size, output_size);

    const auto input = test::random_bf16({ 1, inner_size }, backend);
    const auto weights = test::random_bf16({ output_size, inner_size }, backend);
    auto output = Tensor::empty({ 1, output_size }, types::DType::BF16, backend);
    const auto expected = test::reference::matmul(input, weights);

    backend.matmul(input, weights, output);
    const auto output_host = test::to_host(output);
    CHECK_THAT(output_host, test::WithinRel(expected, 0.01f));
}

TEST_CASE("gemv benchmark", "[benchmark][gemv]") {
    using namespace inference;

    const auto device = GENERATE(types::Device::CPU, types::Device::CUDA);
    CAPTURE(types::device_to_string(device));
    auto& backend = Backend::select_backend(device);

    constexpr std::size_t inner_size = 1024;
    constexpr std::size_t output_size = 1024;
    const auto input = test::random_bf16({ 1, inner_size }, backend);
    const auto weights = test::random_bf16({ output_size, inner_size }, backend);
    auto output = Tensor::empty({ 1, output_size }, types::DType::BF16, backend);

    BENCHMARK(types::device_to_string(device) + " GEMV") {
        backend.matmul(input, weights, output);
        if (device == types::Device::CUDA) {
            if (const auto error = cudaDeviceSynchronize(); error != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(error));
            }
        }
    };
}
