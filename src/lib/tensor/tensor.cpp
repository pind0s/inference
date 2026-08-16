#include "tensor.hpp"
#include "backend/backend.hpp"
#include "storage.hpp"
#include "tensor_shape.hpp"
#include "types/device.hpp"
#include "types/dtype.hpp"
#include <cstddef>
#include <cstring>
#include <cuda_runtime_api.h>
#include <memory>
#include <span>
#include <stdexcept>

namespace inference {
    Tensor::Tensor(const TensorShape& shape, const types::DType dtype, Backend& backend)
        : storage_(std::make_shared<Storage>(backend, shape.size() * types::dtype_byte_size(dtype))),
          shape_(shape),
          dtype_(dtype) { }

    Tensor Tensor::empty(const TensorShape& shape, const types::DType dtype, Backend& backend) {
        return { shape, dtype, backend };
    }

    Tensor Tensor::from_host_bytes(const std::span<const std::byte> source, const TensorShape& shape, const types::DType dtype) {
        if (source.size_bytes() != shape.size() * types::dtype_byte_size(dtype)) {
            throw std::invalid_argument("tensor shape and dtype do not match the supplied data size");
        }

        auto& host_backend = Backend::get_backend<types::Device::CPU>();
        auto result = empty(shape, dtype, host_backend);
        std::memcpy(result.storage_->ptr, source.data(), source.size_bytes());
        return result;
    }

    Tensor Tensor::host_to_device() const {
        if (!is_cpu()) {
            throw std::invalid_argument("host_to_device called on tensor that is not on host");
        }

        auto& device_backend = Backend::get_backend<types::Device::CUDA>();
        auto result = empty(shape_, dtype_, device_backend);
        if (const auto error = cudaMemcpy(result.storage_->ptr, storage_->ptr, size_bytes(), cudaMemcpyHostToDevice); error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }

        return result;
    }

    Tensor Tensor::device_to_host() const {
        if (!is_cuda()) {
            throw std::invalid_argument("device_to_host called on tensor that is not on device");
        }

        auto& host_backend = Backend::get_backend<types::Device::CPU>();
        auto result = empty(shape_, dtype_, host_backend);
        if (const auto error = cudaMemcpy(result.storage_->ptr, storage_->ptr, size_bytes(), cudaMemcpyDeviceToHost); error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }

        return result;
    }
} // namespace inference
