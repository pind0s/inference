#include "reference/matmul.hpp"
#include "common/backend_test.hpp"
#include "common/random.hpp"
#include "common/tolerance.hpp"
#include "tensor/tensor.hpp"
#include <array>
#include <cmath>
#include <ranges>

namespace test {
    struct MatmulShape {
        std::size_t rows;
        std::size_t columns;
        std::size_t inner_size;
    };

    class MatmulTest : public BackendTest { };

    TEST_P(MatmulTest, MatchesReference) {
        using namespace inference;
        static constexpr auto shapes = std::array{
            MatmulShape{ 48, 18992, 1024 }, MatmulShape{ 48, 3072, 1024 }, MatmulShape{ 48, 1024, 3072 }, MatmulShape{ 1, 1, 31 }, MatmulShape{ 3, 5, 33 },
        };

        auto& target_backend = backend();
        auto& cpu_backend = Backend::get_backend<types::Device::CPU>();

        for (const auto [rows, columns, inner_size] : shapes) {
            SCOPED_TRACE(testing::Message() << "shape: " << rows << 'x' << columns << 'x' << inner_size);
            const auto host_input = util::random_bf16_tensor({ rows, inner_size });
            const auto host_weights = util::random_bf16_tensor({ columns, inner_size });
            auto expected = Tensor::empty({ rows, columns }, types::DType::BF16, cpu_backend);
            reference::matmul(host_input.view<cpu::bf16_t>(), host_weights.view<cpu::bf16_t>(), expected.view<cpu::bf16_t>());

            const auto input = host_input.copy_to_backend(target_backend);
            const auto weights = host_weights.copy_to_backend(target_backend);
            auto output = Tensor::empty({ rows, columns }, types::DType::BF16, target_backend);
            target_backend.matmul(input, weights, output);

            const auto actual = output.copy_to_backend(cpu_backend);
            const auto absolute_tolerance = 0.01F * std::sqrt(static_cast<float>(inner_size));
            for (const auto [expected_value, actual_value] : std::views::zip(expected.view<cpu::bf16_t>(), actual.view<cpu::bf16_t>())) {
                ASSERT_NEAR(expected_value, actual_value, util::tolerance_for(expected_value, absolute_tolerance));
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, MatmulTest, testing::Values(inference::types::Device::CPU, inference::types::Device::CUDA), backend_name);
} // namespace test
