#include "cpu/kernels/elementwise.hpp"
#include "util.hpp"

#include <gtest/gtest.h>
#include <ranges>
#include <vector>

class AddTest : public testing::TestWithParam<std::size_t> { };

TEST_P(AddTest, MatchesReference) {
    using namespace inference;

    const auto element_count = GetParam();
    const auto lhs = test::random_bf16_vector(element_count);
    const auto rhs = test::random_bf16_vector(element_count);

    std::vector<__bf16> reference_output(element_count);
    std::vector<__bf16> actual_output(element_count);

    cpu::kernels::reference::add(lhs.data(), rhs.data(), reference_output.data(), element_count);
    cpu::kernels::add(lhs.data(), rhs.data(), actual_output.data(), element_count);

    for (const auto [reference, actual] : std::views::zip(reference_output, actual_output)) {
        EXPECT_FLOAT_EQ(static_cast<float>(reference), static_cast<float>(actual));
    }
}

constexpr auto element_counts = { std::size_t{ 0 },  std::size_t{ 1 },  std::size_t{ 15 }, std::size_t{ 16 },    std::size_t{ 17 },
                                  std::size_t{ 31 }, std::size_t{ 32 }, std::size_t{ 33 }, std::size_t{ 151936 } };
INSTANTIATE_TEST_SUITE_P(ElementCounts, AddTest, testing::ValuesIn(element_counts));
