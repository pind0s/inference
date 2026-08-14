#pragma once
#include "cpu/kernels/sampling.hpp"
#include "tensor/tensor.hpp"
#include "types/device.hpp"
#include <stdexcept>
#include <utility>

namespace inference::ops {
    inline std::size_t argmax(const Tensor& logits) {
        switch (logits.device()) {
        case types::Device::CPU:
            return cpu::kernels::argmax_avx512bf16(logits.data<__bf16>(), logits.num_elems());
        case types::Device::CUDA:
            throw std::runtime_error("cuda not implemented");
        }

        std::unreachable();
    }
} // namespace inference::ops
