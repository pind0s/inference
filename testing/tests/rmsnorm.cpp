#include "reference/rmsnorm.hpp"
#include "backend/backend.hpp"
#include "types/device.hpp"
#include "util/random.hpp"
#include "util/tensor_matchers.hpp"
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <cstddef>

constexpr float epsilon = 1e-6f;

TEST_CASE("rmsnorm matches the reference", "[backend][rmsnorm]") {
    using namespace inference;

    const auto device = GENERATE(types::Device::CPU, types::Device::CUDA);
    CAPTURE(types::device_to_string(device));
    auto& backend = Backend::select_backend(device);

    const auto hidden_size = GENERATE(128uz, 256uz, 1024uz);
    CAPTURE(hidden_size);

    auto input = test::random_bf16({ hidden_size }, backend);
    const auto weight = test::random_bf16({ hidden_size }, backend, { 1 });
    auto output = Tensor::empty(input.shape(), types::DType::BF16, backend);
    const auto expected = test::reference::rmsnorm(input, weight, epsilon);

    backend.rmsnorm(input, weight, output, epsilon);
    CHECK_THAT(output.to_host(), test::WithinRel(expected, 0.01f));
}

TEST_CASE("rmsnorm add matches separate add and rmsnorm", "[backend][rmsnorm_add]") {
    using namespace inference;

    const auto device = GENERATE(types::Device::CPU, types::Device::CUDA);
    CAPTURE(types::device_to_string(device));
    auto& backend = Backend::select_backend(device);

    const auto hidden_size = GENERATE(128uz, 256uz, 1024uz);
    CAPTURE(hidden_size);

    auto residual = test::random_bf16({ hidden_size }, backend);
    const auto update = test::random_bf16({ hidden_size }, backend, { 1 });
    const auto weight = test::random_bf16({ hidden_size }, backend, { 2 });
    auto expected_residual = Tensor::empty(residual.shape(), types::DType::BF16, backend);
    auto output = Tensor::empty(residual.shape(), types::DType::BF16, backend);

    backend.add(residual, update, expected_residual);
    const auto expected_output = test::reference::rmsnorm(expected_residual, weight, epsilon);
    backend.rmsnorm_add(residual, update, weight, output, epsilon);
    CHECK_THAT(residual.to_host(), test::WithinRel(expected_residual.to_host(), 0.01f));
    CHECK_THAT(output.to_host(), test::WithinRel(expected_output, 0.01f));
}

TEST_CASE("rmsnorm benchmark", "[benchmark][rmsnorm]") {
    using namespace inference;

    const auto device = GENERATE(types::Device::CPU, types::Device::CUDA);
    CAPTURE(types::device_to_string(device));
    auto& backend = Backend::select_backend(device);

    constexpr std::size_t hidden_size = 1024;
    auto input = test::random_bf16({ hidden_size }, backend);
    const auto weight = test::random_bf16({ hidden_size }, backend);
    auto output = Tensor::empty(input.shape(), types::DType::BF16, backend);

    BENCHMARK(types::device_to_string(device) + " RMSNorm") {
        backend.rmsnorm(input, weight, output, epsilon);
        backend.synchronize();
    };
}

TEST_CASE("rmsnorm add benchmark", "[benchmark][rmsnorm_add]") {
    using namespace inference;

    const auto device = GENERATE(types::Device::CPU, types::Device::CUDA);
    CAPTURE(types::device_to_string(device));
    auto& backend = Backend::select_backend(device);

    constexpr std::size_t hidden_size = 1024;
    auto residual = test::random_bf16({ hidden_size }, backend);
    const auto update = test::random_bf16({ hidden_size }, backend);
    const auto weight = test::random_bf16({ hidden_size }, backend);
    auto output = Tensor::empty(residual.shape(), types::DType::BF16, backend);

    BENCHMARK(types::device_to_string(device) + " RMSNorm Add") {
        backend.rmsnorm_add(residual, update, weight, output, epsilon);
        backend.synchronize();
    };
}
