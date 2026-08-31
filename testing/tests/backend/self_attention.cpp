#include "reference/self_attention.hpp"
#include "tensor/tensor.hpp"
#include "util/backend_test.hpp"
#include "util/copy_tensor_to_host.hpp"
#include "util/random.hpp"
#include "util/tolerance.hpp"
#include <array>
#include <ranges>

namespace test {
    struct SelfAttentionShape {
        std::size_t query_head_count;
        std::size_t key_value_head_count;
        std::size_t head_size;
    };

    class SelfAttentionTest : public BackendTest { };

    TEST_P(SelfAttentionTest, MatchesReference) {
        using namespace inference;
        static constexpr auto position = std::size_t{ 3 };
        static constexpr auto shapes = std::array{
            SelfAttentionShape{ 4, 2, 128 },
            SelfAttentionShape{ 4, 4, 64 },
        };

        auto& target_backend = backend();
        auto& cpu_backend = Backend::get_backend<types::Device::CPU>();

        for (const auto [query_head_count, key_value_head_count, head_size] : shapes) {
            SCOPED_TRACE(testing::Message() << "shape: " << query_head_count << 'x' << key_value_head_count << 'x' << head_size);

            auto query_shape = { query_head_count, head_size };
            auto cache_shape = { position + 1, key_value_head_count, head_size };

            const auto host_query = util::random_bf16_tensor(query_shape);
            const auto host_key = util::random_bf16_tensor(cache_shape);
            const auto host_value = util::random_bf16_tensor(cache_shape);
            auto expected = Tensor::empty(query_shape, types::DType::BF16, cpu_backend);
            reference::self_attention(host_query.view<cpu::bf16_t>(), host_key.view<cpu::bf16_t>(), host_value.view<cpu::bf16_t>(),
                                      expected.view<cpu::bf16_t>(), position);

            const auto query = target_backend.make_tensor(host_query.bytes(), host_query.shape(), host_query.dtype());
            const auto key = target_backend.make_tensor(host_key.bytes(), host_key.shape(), host_key.dtype());
            const auto value = target_backend.make_tensor(host_value.bytes(), host_value.shape(), host_value.dtype());
            auto output = Tensor::empty(query_shape, types::DType::BF16, target_backend);
            target_backend.self_attention(query, key, value, output, position);

            const auto actual = util::copy_tensor_to_host<cpu::bf16_t>(output);
            for (const auto [expected_value, actual_value] : std::views::zip(expected.view<cpu::bf16_t>(), actual)) {
                ASSERT_NEAR(expected_value, actual_value, util::tolerance_for(expected_value));
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, SelfAttentionTest, testing::Values(inference::types::Device::CPU, inference::types::Device::CUDA), backend_name);
} // namespace test
