#include "cpu/kernels/sampling.hpp"
#include "util.hpp"
#include <algorithm>
#include <gtest/gtest.h>

class ArgmaxTest : public testing::TestWithParam<std::size_t> { };

TEST_P(ArgmaxTest, MatchesReference) {
    using namespace inference;
    const auto element_count = GetParam();

    const auto logits = test::random_bf16_vector(element_count);

    const auto expected = std::max_element(logits.data(), logits.data() + element_count) - logits.data();
    const auto actual = cpu::kernels::argmax_avx512bf16(logits.data(), element_count);

    EXPECT_EQ(expected, actual);
}

constexpr auto element_counts = { std::size_t{ 1 },  std::size_t{ 15 }, std::size_t{ 16 }, std::size_t{ 17 },
                                  std::size_t{ 31 }, std::size_t{ 32 }, std::size_t{ 33 }, std::size_t{ 151936 } };
INSTANTIATE_TEST_SUITE_P(ElementCounts, ArgmaxTest, testing::ValuesIn(element_counts));
