#pragma once
#include "backend/util/rope_cache.hpp"
#include "tensor/tensor.hpp"
#include <cstring>
#include <cuda_runtime_api.h>
#include <stdexcept>

namespace test {
    [[nodiscard]] inline inference::Tensor to_host(const inference::Tensor& tensor) {
        auto& cpu = inference::Backend::get_backend<inference::types::Device::CPU>();
        auto result = inference::Tensor::empty(tensor.shape(), tensor.dtype(), cpu);
        if (tensor.size_bytes() == 0) {
            return result;
        }

        if (tensor.is_cpu()) {
            std::memcpy(result.bytes().data(), tensor.bytes().data(), tensor.size_bytes());
        } else {
            if (const auto error = cudaDeviceSynchronize(); error != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(error));
            }
            if (const auto error = cudaMemcpy(result.bytes().data(), tensor.bytes().data(), tensor.size_bytes(), cudaMemcpyDeviceToHost);
                error != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(error));
            }
        }
        return result;
    }

    [[nodiscard]] inline inference::backend::RopeCache to_host(const inference::backend::RopeCache& cache) {
        return { .cosine = to_host(cache.cosine), .sine = to_host(cache.sine) };
    }

} // namespace test
