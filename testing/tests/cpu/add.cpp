#include "backend/cpu/kernels/elementwise.hpp"
#include "util.hpp"
#include <common/random.hpp>
#include <gtest/gtest.h>
#include <ranges>
#include <vector>

class AddTest : public testing::TestWithParam<std::size_t> { };

TEST_P(AddTest, MatchesReference) {
    using namespace inference;

    const auto element_count = GetParam();
    auto& cpu_backend = Backend::get_backend<types::Device::CPU>();

    const auto shape = { element_count };

    const auto lhs = test::util::cpu::random_bf16_tensor(shape);
    const auto rhs = test::util::cpu::random_bf16_tensor(shape);
    const auto lhs_view = lhs.view<cpu::bf16_t>();
    const auto rhs_view = rhs.view<cpu::bf16_t>();

    auto expected_tensor = Tensor::empty(shape, types::DType::BF16, cpu_backend);
    auto actual_tensor = Tensor::empty(shape, types::DType::BF16, cpu_backend);
    auto expected_view = expected_tensor.view<cpu::bf16_t>();
    auto actual_view = actual_tensor.view<cpu::bf16_t>();

    for (std::size_t index = 0; index < element_count; ++index) {
        expected_view[index] = lhs_view[index].to_float() + rhs_view[index].to_float();
    }
    cpu::kernels::add(lhs_view, rhs_view, actual_tensor.view<cpu::bf16_t>());

    for (const auto [expected, actual] : std::views::zip(expected_view, actual_view)) {
        EXPECT_FLOAT_EQ(expected, actual);
    }
}

constexpr auto element_counts = { std::size_t{ 0 },  std::size_t{ 1 },  std::size_t{ 15 }, std::size_t{ 16 },    std::size_t{ 17 },
    std::size_t{ 31 }, std::size_t{ 32 }, std::size_t{ 33 }, std::size_t{ 151936 },
};
INSTANTIATE_TEST_SUITE_P(ElementCounts, AddTest, testing::ValuesIn(element_counts));
