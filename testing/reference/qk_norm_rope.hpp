#pragma once
#include "rmsnorm.hpp"
#include "rope.hpp"

namespace test::reference {
    [[nodiscard]] inline inference::Tensor norm_rope(const inference::Tensor& input, const inference::Tensor& norm,
                                                     const inference::backend::RopeCache& cache, const std::size_t position, const float epsilon) {
        const auto normalized = rmsnorm(input, norm, epsilon);
        return rope(normalized, cache, position);
    }
} // namespace test::reference
