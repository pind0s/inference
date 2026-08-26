#include "reference/rope.hpp"
#include "backend/rope_cache.hpp"
#include "common/backend_test.hpp"
#include "common/random.hpp"
#include "common/tolerance.hpp"
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
        auto& cpu_backend = Backend::get_backend<types::Device::CPU>();
        const auto cache = make_rope_cache(position + 1, head_size, theta, target_backend);
        auto expected = util::random_bf16_tensor(shape, util::RandomSeed{ 1 });
        auto actual = expected.copy_to_backend(target_backend);

        reference::rope(expected.view<cpu::bf16_t>(), theta, position);
        target_backend.rope(actual, cache, position);

        const auto actual_host = actual.copy_to_backend(cpu_backend);
        for (const auto [expected_value, actual_value] : std::views::zip(expected.view<cpu::bf16_t>(), actual_host.view<cpu::bf16_t>())) {
            ASSERT_NEAR(expected_value, actual_value, util::tolerance_for(expected_value));
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, RopeTest, testing::Values(inference::types::Device::CPU, inference::types::Device::CUDA), backend_name);
} // namespace test
