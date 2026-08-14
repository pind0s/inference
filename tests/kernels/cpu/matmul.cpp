
#include "cpu/kernels/matmul.hpp"

#include "util.hpp"

#include <cmath>
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
    const auto a = test::random_bf16_vector(m * k);
    const auto b = test::random_bf16_vector(n * k);

    std::vector<__bf16> expected_vec(m * n);
    std::vector<__bf16> actual_vec(m * n);

    cpu::kernels::reference::naive_matmul_bf16(a.data(), b.data(), expected_vec.data(), m, n, k);
    cpu::kernels::matmul_avx512bf16(a.data(), b.data(), actual_vec.data(), m, n, k);

    for (const auto [expected, actual] : std::views::zip(expected_vec, actual_vec)) {
        const auto tolerance = 0.01F + (0.01F * std::abs(static_cast<float>(expected)));
        EXPECT_NEAR(expected, actual, tolerance);
    }
}

constexpr auto shapes = {
    MatmulShape{ 48, 18992, 1024 }, MatmulShape{ 48, 3072, 1024 }, MatmulShape{ 48, 1024, 3072 }, MatmulShape{ 1, 1, 31 }, MatmulShape{ 3, 5, 33 },
};
INSTANTIATE_TEST_SUITE_P(Shapes, MatmulGemmTest, testing::ValuesIn(shapes));
