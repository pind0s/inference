#pragma once
#include "tensor/tensor.hpp"
#include <cstring>
#include <cuda_runtime_api.h>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace test::util {
    template <typename T>
    [[nodiscard]] std::vector<T> copy_tensor_to_host(const inference::Tensor& tensor) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (tensor.size_bytes() != tensor.size() * sizeof(T)) {
            throw std::invalid_argument("tensor element type does not match its data size");
        }

        auto values = std::vector<T>(tensor.size());
        if (values.empty()) {
            return values;
        }

        const auto destination = std::as_writable_bytes(std::span{ values });
        if (tensor.is_cpu()) {
            std::memcpy(destination.data(), tensor.bytes().data(), destination.size_bytes());
        } else {
            if (const auto error = cudaDeviceSynchronize(); error != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(error));
            }
            if (const auto error = cudaMemcpy(destination.data(), tensor.bytes().data(), destination.size_bytes(), cudaMemcpyDeviceToHost);
                error != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(error));
            }
        }
        return values;
    }
} // namespace test::util
