#pragma once
#include "allocator/cpu_allocator.hpp"
#include "kv_cache/kv_cache.hpp"

namespace inference {
    struct CpuContext {
        std::shared_ptr<allocator::CpuAllocator> allocator = std::make_shared<allocator::CpuAllocator>();
    };

    struct Context {
        CpuContext cpu_context;
        KVCache kv_cache;
        // cuda context in the future
    };
} // namespace inference
