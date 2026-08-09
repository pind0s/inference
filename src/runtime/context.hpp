#pragma once
#include "allocator/cpu_allocator.hpp"

namespace inference {
    struct CpuContext {
        std::shared_ptr<allocator::CpuAllocator> allocator;
        CpuContext(): allocator(std::make_shared<allocator::CpuAllocator>()) { }
    };

    struct Context {
        CpuContext cpu_context;
        // cuda context in the future

        static Context create_cpu_context() {
            return {};
        }
    };
} // namespace inference
