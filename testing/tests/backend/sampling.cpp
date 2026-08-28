#include "util/backend_test.hpp"
#include "util/random.hpp"
#include <algorithm>
#include <array>

namespace test {
    class ArgmaxTest : public BackendTest { };

    TEST_P(ArgmaxTest, MatchesReference) {
        using namespace inference;
        static constexpr auto element_counts = std::array<std::size_t, 8>{ 1, 15, 16, 17, 31, 32, 33, 151936 };
        auto& target_backend = backend();

        for (const auto element_count : element_counts) {
            SCOPED_TRACE(testing::Message() << "element count: " << element_count);
            const auto host_logits = util::random_bf16_tensor({ element_count });
            const auto logits_values = host_logits.view<cpu::bf16_t>();
            const auto expected = std::max_element(logits_values.data(), logits_values.data() + element_count) - logits_values.data();
            const auto logits = target_backend.make_tensor(host_logits.bytes(), host_logits.shape(), host_logits.dtype());

            EXPECT_EQ(expected, target_backend.argmax(logits));
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, ArgmaxTest, testing::Values(inference::types::Device::CPU, inference::types::Device::CUDA), backend_name);
} // namespace test
