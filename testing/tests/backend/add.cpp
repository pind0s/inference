#include "common/backend_test.hpp"
#include "common/random.hpp"
#include "common/tolerance.hpp"
#include "tensor/tensor.hpp"
#include <array>
#include <ranges>

namespace test {
    class AddTest : public BackendTest { };

    TEST_P(AddTest, MatchesReference) {
        using namespace inference;
        static constexpr auto element_counts = std::array<std::size_t, 6>{ 15, 16, 17, 31, 32, 33 };

        auto& target_backend = backend();
        auto& cpu_backend = Backend::get_backend<types::Device::CPU>();

        for (const auto element_count : element_counts) {
            SCOPED_TRACE(testing::Message() << "element count: " << element_count);
            const TensorShape shape{ element_count };
            const auto host_lhs = util::random_bf16_tensor(shape);
            const auto host_rhs = util::random_bf16_tensor(shape);

            auto expected = Tensor::empty(shape, types::DType::BF16, cpu_backend);
            const auto lhs_values = host_lhs.view<cpu::bf16_t>();
            const auto rhs_values = host_rhs.view<cpu::bf16_t>();
            auto expected_values = expected.view<cpu::bf16_t>();
            for (std::size_t index = 0; index < element_count; ++index) {
                expected_values[index] = lhs_values[index].to_float() + rhs_values[index].to_float();
            }

            const auto lhs = host_lhs.copy_to_backend(target_backend);
            const auto rhs = host_rhs.copy_to_backend(target_backend);
            auto output = Tensor::empty(shape, types::DType::BF16, target_backend);
            target_backend.add(lhs, rhs, output);

            const auto actual = output.copy_to_backend(cpu_backend);
            for (const auto [expected_value, actual_value] : std::views::zip(expected.view<cpu::bf16_t>(), actual.view<cpu::bf16_t>())) {
                ASSERT_NEAR(expected_value, actual_value, util::tolerance_for(expected_value));
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, AddTest, testing::Values(inference::types::Device::CPU, inference::types::Device::CUDA), backend_name);
} // namespace test
