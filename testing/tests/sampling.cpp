#include "reference/sampling.hpp"
#include "backend/backend.hpp"
#include "types/device.hpp"
#include "util/random.hpp"
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstddef>

TEST_CASE("sampling argmax matches the reference", "[backend][sampling]") {
    using namespace inference;

    const auto device = GENERATE(types::Device::CPU, types::Device::CUDA);
    CAPTURE(types::device_to_string(device));
    auto& backend = Backend::select_backend(device);

    const auto elem_count = GENERATE(1uz, 15uz, 16uz, 17uz, 255uz, 256uz, 257uz);
    CAPTURE(elem_count);
    const auto logits = test::random_bf16({ elem_count }, backend);
    const auto expected = test::reference::argmax(logits);

    CHECK(backend.argmax(logits) == expected);
}

TEST_CASE("sampling benchmark", "[benchmark][sampling]") {
    using namespace inference;

    const auto device = GENERATE(types::Device::CPU, types::Device::CUDA);
    CAPTURE(types::device_to_string(device));
    auto& backend = Backend::select_backend(device);

    constexpr std::size_t vocab_size = 151'936;
    const auto logits = test::random_bf16({ vocab_size }, backend);

    BENCHMARK(types::device_to_string(device) + " Argmax") {
        return backend.argmax(logits);
    };
}
