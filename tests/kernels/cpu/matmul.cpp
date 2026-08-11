
#include <cmath>
#include <gtest/gtest.h>
#include <ranges>
#include <vector>

#include "cpu/bf16.hpp"
#include "cpu/kernels/matmul.hpp"
#include "reference/cpu_matmul.hpp"
#include "util.hpp"

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

    std::vector<cpu::bf16_t> reference_output(m * n);
    std::vector<cpu::bf16_t> actual_output(m * n);

    test::reference::cpu::naive_matmul_bf16(a.data(), b.data(), reference_output.data(), m, n, k);
    cpu::kernels::matmul_avx512bf16(a.data(), b.data(), actual_output.data(), m, n, k);

    for (const auto [reference, actual] : std::views::zip(reference_output, actual_output)) {
        const auto tolerance = 0.01F + (0.01F * std::abs(actual.to_float()));
        EXPECT_NEAR(reference.to_float(), actual.to_float(), tolerance);
    }
}

constexpr auto shapes = {
    MatmulShape{48, 18992, 1024},
    MatmulShape{48, 3072, 1024},
    MatmulShape{48, 1024, 3072},
    MatmulShape{1, 1, 31},
    MatmulShape{3, 5, 33},
};
INSTANTIATE_TEST_SUITE_P(Shapes, MatmulGemmTest, testing::ValuesIn(shapes));
