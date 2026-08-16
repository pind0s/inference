#include "backend/cpu/kernels/self_attention.hpp"
#include "reference/self_attention.hpp"
#include "util.hpp"
#include <common/random.hpp>
#include <gtest/gtest.h>
#include <ranges>
#include <vector>

namespace inference::cpu::kernels {
    TEST(SelfAttention, OptimizedMatchesReferenceWithRankedViews) {
        constexpr std::size_t position = 3;
        constexpr std::size_t query_head_count = 4;
        constexpr std::size_t key_value_head_count = 2;
        constexpr std::size_t head_size = 33;

        constexpr TensorShape query_shape{ query_head_count, head_size };
        constexpr TensorShape cache_shape{ position + 1, key_value_head_count, head_size };
        const auto query = test::util::cpu::random_bf16_tensor(query_shape);
        const auto key = test::util::cpu::random_bf16_tensor(cache_shape);
        const auto value = test::util::cpu::random_bf16_tensor(cache_shape);
        std::vector<bf16_t> expected(query_shape.size());
        std::vector<bf16_t> actual(query_shape.size());

        test::reference::cpu::self_attention(query.view<bf16_t>(), key.view<bf16_t>(), value.view<bf16_t>(), TensorView{ expected.data(), query_shape },
                                             position);
        self_attention(query.view<bf16_t>(), key.view<bf16_t>(), value.view<bf16_t>(), TensorView{ actual.data(), query_shape }, position);

        for (const auto [reference, optimized] : std::views::zip(expected, actual)) {
            EXPECT_NEAR(reference, optimized, test::tolerance_for(reference));
        }
    }
} // namespace inference::cpu::kernels
