#include "backend/backend.hpp"
#include "backend/cpu/kernels/elementwise.hpp"
#include "backend/cpu/kernels/embedding.hpp"
#include "backend/cpu/kernels/kv_cache.hpp"
#include "backend/cpu/kernels/matmul.hpp"
#include "backend/cpu/kernels/rmsnorm.hpp"
#include "backend/cpu/kernels/rope.hpp"
#include "backend/cpu/kernels/sampling.hpp"
#include "backend/cpu/kernels/self_attention.hpp"
#include "backend/rope_cache.hpp"
#include "tensor/tensor.hpp"

namespace inference {
    namespace cpu {

        class CpuBackend : public Backend {
        public:
            [[nodiscard]] types::Device device() const override {
                return types::Device::CPU;
            }

            [[nodiscard]] void* allocate(std::size_t size_bytes) override {
                return operator new(size_bytes, std::align_val_t{ alignment });
            }

            void deallocate(void* pointer, std::size_t bytes) noexcept override {
                operator delete(pointer, bytes, std::align_val_t{ alignment });
            }

            void embedding(const types::TokenId token_id, const Tensor& weights, Tensor& output) override {
                assert(output.dtype() == types::DType::BF16);
                kernels::embedding(token_id, weights.view<bf16_t>(), output.view<bf16_t>());
            }

            void matmul(const Tensor& input, const Tensor& weights, Tensor& output) override {
                assert(output.dtype() == types::DType::BF16);
                kernels::matmul_avx512bf16(input.view<bf16_t>(), weights.view<bf16_t>(), output.view<bf16_t>());
            }

            void rmsnorm(const Tensor& input, const Tensor& weight, Tensor& output, const float epsilon) override {
                assert(output.dtype() == types::DType::BF16);
                kernels::rmsnorm(input.view<bf16_t>(), weight.view<bf16_t>(), output.view<bf16_t>(), epsilon);
            }

            void rope(Tensor& values, const RopeCache& cache, const std::size_t position) override {
                assert(values.dtype() == types::DType::BF16);
                kernels::rope(values.view<bf16_t>(), cache.view(), position);
            }

            void kv_cache_update(const Tensor& key, const Tensor& value, Tensor& cached_key, Tensor& cached_value,
                                 const std::size_t token_offset) override {
                assert(key.dtype() == types::DType::BF16);
                kernels::kv_cache_update(key.view<bf16_t>(), value.view<bf16_t>(), cached_key.view<bf16_t>(), cached_value.view<bf16_t>(), token_offset);
            }

            void self_attention(const Tensor& query, const Tensor& key, const Tensor& value, Tensor& output, const std::size_t position) override {
                assert(output.dtype() == types::DType::BF16);
                kernels::self_attention(query.view<bf16_t>(), key.view<bf16_t>(), value.view<bf16_t>(), output.view<bf16_t>(), position);
            }

            void add(const Tensor& left, const Tensor& right, Tensor& output) override {
                assert(output.dtype() == types::DType::BF16);
                kernels::add(left.view<bf16_t>(), right.view<bf16_t>(), output.view<bf16_t>());
            }

            void silu(const Tensor& gate, const Tensor& up, Tensor& output) override {
                assert(output.dtype() == types::DType::BF16);
                kernels::silu(gate.view<bf16_t>(), up.view<bf16_t>(), output.view<bf16_t>());
            }

            [[nodiscard]] types::TokenId argmax(const Tensor& logits) override {
                assert(logits.dtype() == types::DType::BF16);
                return kernels::argmax_avx512bf16(logits.view<bf16_t>());
            }

        private:
            static constexpr std::size_t alignment = 64;
        };
    } // namespace cpu

    template <>
    Backend& Backend::get_backend<types::Device::CPU>() {
        static auto backend = cpu::CpuBackend();
        return backend;
    }
} // namespace inference
