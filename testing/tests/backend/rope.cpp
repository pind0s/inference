#include "reference/rope.hpp"
#include "backend/util/rope_cache.hpp"
#include "util/backend_test.hpp"
#include "util/copy_tensor_to_host.hpp"
#include "util/random.hpp"
#include "util/tolerance.hpp"
#include <ranges>

namespace test {
    class RopeTest : public BackendTest { };

    TEST_P(RopeTest, MatchesReference) {
        using namespace inference;
        constexpr std::size_t head_count = 8;
        constexpr std::size_t head_size = 128;
        constexpr std::size_t position = 50;
        constexpr auto theta = 1'000'000.0F;
        constexpr TensorShape shape{ head_count, head_size };

        auto& target_backend = backend();
        const auto cache = make_rope_cache(position + 1, head_size, theta, target_backend);
        auto expected = util::random_bf16_tensor(shape, util::RandomSeed{ 1 });
        auto actual = target_backend.make_tensor(expected.bytes(), expected.shape(), expected.dtype());

        reference::rope(expected.view<cpu::bf16_t>(), theta, position);
        target_backend.rope(actual, cache, position);

        const auto actual_values = util::copy_tensor_to_host<cpu::bf16_t>(actual);
        for (const auto [expected_value, actual_value] : std::views::zip(expected.view<cpu::bf16_t>(), actual_values)) {
            ASSERT_NEAR(expected_value, actual_value, util::tolerance_for(expected_value));
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, RopeTest, testing::Values(inference::types::Device::CPU, inference::types::Device::CUDA), backend_name);
} // namespace test
