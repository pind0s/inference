#include "tensor.hpp"
#include "storage.hpp"
#include "tensor_shape.hpp"
#include "types/dtype.hpp"
#include <memory>

namespace inference {
    Tensor::Tensor(const TensorShape& shape, const types::DType dtype, Backend& backend)
        : storage_(std::make_shared<Storage>(backend, shape.size() * types::dtype_byte_size(dtype))),
          shape_(shape),
          dtype_(dtype) { }

    Tensor Tensor::empty(const TensorShape& shape, const types::DType dtype, Backend& backend) {
        return { shape, dtype, backend };
    }
} // namespace inference
