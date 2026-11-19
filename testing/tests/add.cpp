#include "reference/add.hpp"
#include "backend/backend.hpp"
#include "types/device.hpp"
#include "util/random.hpp"
#include "util/tensor_matchers.hpp"
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <cstddef>

TEST_CASE("add matches the reference", "[backend][add]") {
    using namespace inference;

    const auto device = GENERATE(types::Device::CPU, types::Device::CUDA);
    CAPTURE(types::device_to_string(device));
    auto& backend = Backend::select_backend(device);

    const auto elem_count = GENERATE(1uz, 15uz, 16uz, 17uz, 255uz, 256uz, 257uz);
    CAPTURE(elem_count);

    const auto lhs = test::random_bf16({ elem_count }, backend);
    const auto rhs = test::random_bf16({ elem_count }, backend, { 1 });
    auto output = Tensor::empty({ elem_count }, types::DType::BF16, backend);
    const auto expected = test::reference::add(lhs, rhs);

    backend.add(lhs, rhs, output);
    CHECK_THAT(output.to_host(), test::WithinRel(expected, 0.01f));
}

TEST_CASE("add benchmark", "[benchmark][add]") {
    using namespace inference;

    const auto device = GENERATE(types::Device::CPU, types::Device::CUDA);
    CAPTURE(types::device_to_string(device));
    auto& backend = Backend::select_backend(device);

    constexpr std::size_t elem_count = 1024;
    const auto lhs = test::random_bf16({ elem_count }, backend);
    const auto rhs = test::random_bf16({ elem_count }, backend);
    auto output = Tensor::empty({ elem_count }, types::DType::BF16, backend);

    BENCHMARK(types::device_to_string(device) + " Add") {
        backend.add(lhs, rhs, output);
        backend.synchronize();
    };
}
