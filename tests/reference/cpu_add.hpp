#pragma once
#include <cstddef>

#include "cpu/bf16.hpp"

namespace test::reference::cpu {
    inline void add(const inference::cpu::bf16_t* __restrict lhs, const inference::cpu::bf16_t* __restrict rhs,
                    inference::cpu::bf16_t* __restrict out, const std::size_t N) {
        for (std::size_t index = 0; index < N; ++index) {
            out[index] = inference::cpu::bf16_t::from_float(lhs[index].to_float() + rhs[index].to_float());
        }
    }
} // namespace test::reference::cpu
