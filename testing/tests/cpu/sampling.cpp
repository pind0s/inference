#include "backend/cpu/kernels/sampling.hpp"
#include "util.hpp"
#include <algorithm>
#include <common/random.hpp>
#include <gtest/gtest.h>

class ArgmaxTest : public testing::TestWithParam<std::size_t> { };

TEST_P(ArgmaxTest, MatchesReference) {
    using namespace inference;
    const auto element_count = GetParam();
    const TensorShape shape{ element_count };

    const auto logits = test::util::cpu::random_bf16_tensor(shape);
    const auto logits_view = logits.view<cpu::bf16_t>();

    const auto expected = std::max_element(logits_view.data(), logits_view.data() + element_count) - logits_view.data();
    const auto actual = cpu::kernels::argmax_avx512bf16(logits_view);

    EXPECT_EQ(expected, actual);
}

constexpr auto element_counts = { std::size_t{ 1 },  std::size_t{ 15 }, std::size_t{ 16 }, std::size_t{ 17 },
                                  std::size_t{ 31 }, std::size_t{ 32 }, std::size_t{ 33 }, std::size_t{ 151936 } };
INSTANTIATE_TEST_SUITE_P(ElementCounts, ArgmaxTest, testing::ValuesIn(element_counts));
