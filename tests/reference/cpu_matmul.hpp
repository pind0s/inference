#pragma once
#include "cpu/bf16.hpp"

namespace test::reference::cpu {
    using namespace inference::cpu;
    // todo not sure about putting this here, but for now it works.
    inline void naive_matmul_bf16(const bf16_t* __restrict a, const bf16_t* __restrict b, bf16_t* __restrict output, const std::size_t M,
                                  const std::size_t N, const std::size_t K) noexcept {

        for (std::size_t row = 0; row < M; ++row) {
            for (std::size_t column = 0; column < N; ++column) {
                float sum = 0.0F;
                for (std::size_t inner = 0; inner < K; ++inner) {
                    sum += a[row * K + inner].to_float() * b[column * K + inner].to_float();
                }
                output[row * N + column] = bf16_t::from_float(sum);
            }
        }
    }

} // namespace test::reference::cpu
