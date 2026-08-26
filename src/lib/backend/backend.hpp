#pragma once
#include "types/device.hpp"
#include "types/token.hpp"
#include <cstddef>

namespace inference {
    struct RopeCache;
    class Tensor;

    class Backend {
    public:
        virtual ~Backend() = default;

        [[nodiscard]] virtual types::Device device() const = 0;
        [[nodiscard]] virtual void* allocate(std::size_t size_bytes) = 0;
        virtual void deallocate(void* pointer, std::size_t size_bytes) noexcept = 0;
        virtual void copy(void* destination, const void* source, std::size_t size_bytes, types::Device source_device,
                          types::Device destination_device) = 0;

        virtual void embedding(types::TokenId token_id, const Tensor& weights, Tensor& output) = 0;
        virtual void matmul(const Tensor& input, const Tensor& weights, Tensor& output) = 0;
        virtual void rmsnorm(const Tensor& input, const Tensor& weight, Tensor& output, float epsilon) = 0;
        virtual void rope(Tensor& values, const RopeCache& cache, std::size_t position) = 0;
        virtual void kv_cache_update(const Tensor& key, const Tensor& value, Tensor& cached_key, Tensor& cached_value, std::size_t token_offset) = 0;
        virtual void self_attention(const Tensor& query, const Tensor& key, const Tensor& value, Tensor& output, std::size_t position) = 0;
        virtual void add(const Tensor& left, const Tensor& right, Tensor& output) = 0;
        virtual void silu(const Tensor& gate, const Tensor& up, Tensor& output) = 0;
        [[nodiscard]] virtual types::TokenId argmax(const Tensor& logits) = 0;

        template <types::Device D>
        static Backend& get_backend();
    };

    template <>
    Backend& Backend::get_backend<types::Device::CPU>();

    template <>
    Backend& Backend::get_backend<types::Device::CUDA>();

} // namespace inference
