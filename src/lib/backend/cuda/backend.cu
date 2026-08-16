#include "backend/backend.hpp"
#include "backend/cuda/kernels/elementwise.cuh"
#include "backend/cuda/kernels/embedding.cuh"
#include "backend/cuda/kernels/flash_attention.cuh"
#include "backend/cuda/kernels/matmul.cuh"
#include "backend/cuda/kernels/rmsnorm.cuh"
#include "backend/cuda/kernels/rope.cuh"
#include "backend/rope_cache.hpp"
#include "tensor/tensor.hpp"
#include "types/device.hpp"
#include <cublas_v2.h>
#include <cuda/algorithm>
#include <cuda/memory_resource>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

namespace inference {
    namespace gpu {
        class CudaBackend : public Backend {
        public:
            CudaBackend(): device_(cuda::devices[0]), stream_(device_), memory_pool_(device_) {
                cublasCreate(&cublas_handle_);
                cublasSetStream(cublas_handle_, stream_.get());
            }

            ~CudaBackend() override {
                cublasDestroy(cublas_handle_);
            }

            [[nodiscard]] types::Device device() const override {
                return types::Device::CUDA;
            }

            [[nodiscard]] void* allocate(std::size_t size_bytes) override {
                return memory_pool_.allocate(stream_, size_bytes);
            }

            void deallocate(void* pointer, std::size_t size_bytes) noexcept override {
                memory_pool_.deallocate(stream_, pointer, size_bytes);
            }

            void embedding(const types::TokenId token_id, const Tensor& weights, Tensor& output) override {
                assert(output.dtype() == types::DType::BF16);
                kernels::embedding(stream_, weights.view<__nv_bfloat16>(), output.view<__nv_bfloat16>(), token_id);
            }

            void matmul(const Tensor& input, const Tensor& weights, Tensor& output) override {
                assert(output.dtype() == types::DType::BF16);
                kernels::matmul_cublas_bf16(cublas_handle_, input.view<__nv_bfloat16>(), weights.view<__nv_bfloat16>(), output.view<__nv_bfloat16>());
            }

            void rmsnorm(const Tensor& input, const Tensor& weight, Tensor& output, const float epsilon) override {
                assert(output.dtype() == types::DType::BF16);
                kernels::rmsnorm(stream_, input.view<__nv_bfloat16>(), weight.view<__nv_bfloat16>(), output.view<__nv_bfloat16>(), epsilon);
            }

            void rope(Tensor& values, const RopeCache& cache, const std::size_t position) override {
                assert(values.dtype() == types::DType::BF16);
                kernels::rope(stream_, values.view<__nv_bfloat16>(), cache.view(), position);
            }

            void kv_cache_update(const Tensor& key, const Tensor& value, Tensor& cached_key, Tensor& cached_value,
                                 const std::size_t token_offset) override {
                assert(key.dtype() == types::DType::BF16);
                const auto offset_bytes = token_offset * key.size_bytes();
                const auto key_bytes = key.view<std::byte>();
                const auto value_bytes = value.view<std::byte>();
                const auto cached_key_bytes = cached_key.view<std::byte>();
                const auto cached_value_bytes = cached_value.view<std::byte>();
                cuda::copy_bytes(stream_, cuda::std::span{ key_bytes.data(), key.size_bytes() },
                                 cuda::std::span{ cached_key_bytes.data() + offset_bytes, key.size_bytes() });
                cuda::copy_bytes(stream_, cuda::std::span{ value_bytes.data(), value.size_bytes() },
                                 cuda::std::span{ cached_value_bytes.data() + offset_bytes, value.size_bytes() });
            }

            void self_attention(const Tensor& query, const Tensor& key, const Tensor& value, Tensor& output, const std::size_t position) override {
                assert(output.dtype() == types::DType::BF16);
                kernels::flash_attention(stream_, query.view<__nv_bfloat16>(), key.view<__nv_bfloat16>(), value.view<__nv_bfloat16>(),
                                         output.view<__nv_bfloat16>(), position);
            }

            void add(const Tensor& left, const Tensor& right, Tensor& output) override {
                assert(output.dtype() == types::DType::BF16);
                kernels::add(stream_, left.view<__nv_bfloat16>(), right.view<__nv_bfloat16>(), output.view<__nv_bfloat16>());
            }

            void silu(const Tensor& gate, const Tensor& up, Tensor& output) override {
                assert(output.dtype() == types::DType::BF16);
                kernels::silu(stream_, gate.view<__nv_bfloat16>(), up.view<__nv_bfloat16>(), output.view<__nv_bfloat16>());
            }

            [[nodiscard]] types::TokenId argmax(const Tensor& logits) override {
                assert(logits.dtype() == types::DType::BF16);
                const auto first = thrust::device_pointer_cast(logits.view<__nv_bfloat16>().data());
                const auto maximum = thrust::max_element(thrust::cuda::par.on(stream_.get()), first, first + logits.size());
                return static_cast<types::TokenId>(maximum - first);
            }

        private:
            cuda::device_ref device_;
            cuda::stream stream_;
            cuda::device_memory_pool memory_pool_;
            cublasHandle_t cublas_handle_{};
        };
    } // namespace gpu

    template <>
    Backend& Backend::get_backend<types::Device::CUDA>() {
        static auto backend = gpu::CudaBackend();
        return backend;
    }
} // namespace inference
