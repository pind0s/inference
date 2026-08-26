#include "tensor.hpp"
#include "backend/backend.hpp"
#include "storage.hpp"
#include "tensor_shape.hpp"
#include "types/device.hpp"
#include "types/dtype.hpp"
#include <cstddef>
#include <cstring>
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

    Tensor Tensor::copy_to_backend(Backend& backend) const {
        auto result = empty(shape_, dtype_, backend);
        if (size_bytes() == 0) {
            return result;
        }

        auto& source_backend = storage_->backend;
        auto& copy_backend = is_cuda() ? source_backend : backend;
        copy_backend.copy(result.storage_->ptr, storage_->ptr, size_bytes(), source_backend.device(), backend.device());
        return result;
    }

    Tensor Tensor::host_to_device() const {
        if (!is_cpu()) {
            throw std::invalid_argument("host_to_device called on tensor that is not on host");
        }

        return copy_to_backend(Backend::get_backend<types::Device::CUDA>());
    }

    Tensor Tensor::device_to_host() const {
        if (!is_cuda()) {
            throw std::invalid_argument("device_to_host called on tensor that is not on device");
        }

        return copy_to_backend(Backend::get_backend<types::Device::CPU>());
    }
} // namespace inference
