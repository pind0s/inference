#include "backend/cpu/kernels/rope.hpp"
#include "backend/rope_cache.hpp"
#include "common/random.hpp"
#include "reference/rope.hpp"
#include "util.hpp"
#include <gtest/gtest.h>

namespace test {
    TEST(CpuRopeTest, MatchesCpuReference) {
        using namespace inference;
        constexpr std::size_t head_count = 8;
        constexpr std::size_t head_size = 128;
        constexpr std::size_t position = 50;
        constexpr auto theta = 1'000'000.0f;
        constexpr TensorShape shape(head_count, head_size);
        constexpr auto seed = util::RandomSeed(1);

        auto& backend = Backend::get_backend<types::Device::CPU>();
        auto rope_cache = make_rope_cache(position + 1, head_size, theta, backend);
        auto expected = util::cpu::random_bf16_tensor(shape, seed);
        auto actual = util::cpu::random_bf16_tensor(shape, seed);

        reference::cpu::rope(expected.view<cpu::bf16_t>(), theta, position);
        cpu::kernels::rope(actual.view<cpu::bf16_t>(), rope_cache.view(), position);

        auto expected_view = expected.view<cpu::bf16_t>();
        const auto actual_view = actual.view<cpu::bf16_t>();
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const auto expected_value = expected_view[index].to_float();
            const auto actual_value = actual_view[index].to_float();
            EXPECT_NEAR(actual_value, expected_value, test::tolerance_for(expected_value)) << "index " << index;
        }
    }
} // namespace test
