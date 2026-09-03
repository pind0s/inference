#include "reference/rope.hpp"
#include "backend/util/rope_cache.hpp"
#include "reference/rmsnorm.hpp"
#include "util/backend_test.hpp"
#include "util/copy_tensor_to_host.hpp"
#include "util/random.hpp"
#include "util/tolerance.hpp"
#include <ranges>

namespace test {
    class RopeTest : public BackendTest { };
    class QkNormRopeTest : public BackendTest { };

    TEST_P(RopeTest, MatchesReference) {
        using namespace inference;
        constexpr std::size_t head_count = 8;
        constexpr std::size_t head_size = 128;
        constexpr std::size_t position = 50;
        constexpr auto theta = 1'000'000.0F;
        constexpr TensorShape shape{ head_count, head_size };

        auto& target_backend = backend();
        auto& cpu_backend = Backend::get_backend<types::Device::CPU>();
        const auto reference_cache = make_rope_cache(position + 1, head_size, theta, cpu_backend);
        const auto target_cache = make_rope_cache(position + 1, head_size, theta, target_backend);
        auto expected = util::random_bf16_tensor(shape);
        auto actual = target_backend.make_tensor(expected.bytes(), expected.shape(), expected.dtype());

        reference::rope(expected.view<cpu::bf16_t>(), reference_cache.view(), position);
        target_backend.rope(actual, target_cache, position);

        const auto actual_values = util::copy_tensor_to_host<cpu::bf16_t>(actual);
        for (const auto [expected_value, actual_value] : std::views::zip(expected.view<cpu::bf16_t>(), actual_values)) {
            ASSERT_NEAR(expected_value, actual_value, util::tolerance_for(expected_value));
        }
    }

    TEST_P(QkNormRopeTest, MatchesReference) {
        using namespace inference;
        constexpr std::size_t query_head_count = 8;
        constexpr std::size_t key_head_count = 2;
        constexpr std::size_t head_size = 128;
        constexpr std::size_t position = 50;
        constexpr auto theta = 1'000'000.0F;
        constexpr auto epsilon = 1.0e-6F;

        auto& target_backend = backend();
        auto& cpu_backend = Backend::get_backend<types::Device::CPU>();
        const auto reference_cache = make_rope_cache(position + 1, head_size, theta, cpu_backend);
        const auto target_cache = make_rope_cache(position + 1, head_size, theta, target_backend);
        const auto q_norm = util::random_bf16_tensor({ head_size });
        const auto k_norm = util::random_bf16_tensor({ head_size });
        auto expected_query = util::random_bf16_tensor({ query_head_count, head_size });
        auto expected_key = util::random_bf16_tensor({ key_head_count, head_size });
        auto actual_query = target_backend.make_tensor(expected_query.bytes(), expected_query.shape(), expected_query.dtype());
        auto actual_key = target_backend.make_tensor(expected_key.bytes(), expected_key.shape(), expected_key.dtype());
        const auto target_q_norm = target_backend.make_tensor(q_norm.bytes(), q_norm.shape(), q_norm.dtype());
        const auto target_k_norm = target_backend.make_tensor(k_norm.bytes(), k_norm.shape(), k_norm.dtype());

        reference::rmsnorm(expected_query.view<cpu::bf16_t>(), q_norm.view<cpu::bf16_t>(), expected_query.view<cpu::bf16_t>(), epsilon);
        reference::rmsnorm(expected_key.view<cpu::bf16_t>(), k_norm.view<cpu::bf16_t>(), expected_key.view<cpu::bf16_t>(), epsilon);
        reference::rope(expected_query.view<cpu::bf16_t>(), reference_cache.view(), position);
        reference::rope(expected_key.view<cpu::bf16_t>(), reference_cache.view(), position);

        target_backend.qk_norm_rope(actual_query, actual_key, target_q_norm, target_k_norm, target_cache, position, epsilon);

        const auto actual_query_values = util::copy_tensor_to_host<cpu::bf16_t>(actual_query);
        for (const auto [expected_value, actual_value] : std::views::zip(expected_query.view<cpu::bf16_t>(), actual_query_values)) {
            ASSERT_NEAR(expected_value, actual_value, util::tolerance_for(expected_value));
        }

        const auto actual_key_values = util::copy_tensor_to_host<cpu::bf16_t>(actual_key);
        for (const auto [expected_value, actual_value] : std::views::zip(expected_key.view<cpu::bf16_t>(), actual_key_values)) {
            ASSERT_NEAR(expected_value, actual_value, util::tolerance_for(expected_value));
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, RopeTest, testing::Values(inference::types::Device::CPU, inference::types::Device::CUDA), backend_name);
    INSTANTIATE_TEST_SUITE_P(Backends, QkNormRopeTest, testing::Values(inference::types::Device::CPU, inference::types::Device::CUDA), backend_name);
} // namespace test
