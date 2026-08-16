#include "backend/cpu/kernels/matmul.hpp"
#include "reference/matmul.hpp"
#include "util.hpp"
#include <common/random.hpp>
#include <gtest/gtest.h>
#include <ranges>
#include <vector>

struct MatmulShape {
    std::size_t m;
    std::size_t n;
    std::size_t k;
};

class MatmulGemmTest : public testing::TestWithParam<MatmulShape> { };

TEST_P(MatmulGemmTest, MatchesReference) {
    using namespace inference;

    const auto [m, n, k] = GetParam();
    const auto a = test::util::cpu::random_bf16_tensor(TensorShape{ m, k });
    const auto b = test::util::cpu::random_bf16_tensor(TensorShape{ n, k });

    std::vector<cpu::bf16_t> expected_vec(m * n);
    std::vector<cpu::bf16_t> actual_vec(m * n);

    test::reference::cpu::matmul(a.view<cpu::bf16_t>(), b.view<cpu::bf16_t>(), TensorView<cpu::bf16_t>{ expected_vec.data(), TensorShape{ m, n } });
    cpu::kernels::matmul_avx512bf16(a.view<cpu::bf16_t>(), b.view<cpu::bf16_t>(), TensorView<cpu::bf16_t>{ actual_vec.data(), TensorShape{ m, n } });

    for (const auto [expected, actual] : std::views::zip(expected_vec, actual_vec)) {
        EXPECT_NEAR(expected, actual, test::tolerance_for(expected));
    }
}

constexpr auto shapes = {
    MatmulShape{ 48, 18992, 1024 }, MatmulShape{ 48, 3072, 1024 }, MatmulShape{ 48, 1024, 3072 }, MatmulShape{ 1, 1, 31 }, MatmulShape{ 3, 5, 33 },
};
INSTANTIATE_TEST_SUITE_P(Shapes, MatmulGemmTest, testing::ValuesIn(shapes));
