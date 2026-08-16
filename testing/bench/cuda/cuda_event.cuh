#pragma once
#include <cuda_runtime_api.h>

namespace test {
    class CudaEventPair {
    public:
        CudaEventPair() {
            cudaEventCreate(&start_);
            cudaEventCreate(&stop_);
        }

        ~CudaEventPair() {
            cudaEventDestroy(stop_);
            cudaEventDestroy(start_);
        }

        CudaEventPair(const CudaEventPair&) = delete;
        CudaEventPair& operator=(const CudaEventPair&) = delete;
        CudaEventPair(const CudaEventPair&&) = delete;
        CudaEventPair& operator=(const CudaEventPair&&) = delete;

        void record_start(const cuda::stream_ref stream) const {
            cudaEventRecord(start_, stream.get());
        }

        [[nodiscard]] double record_stop_and_elapsed_seconds(const cuda::stream_ref stream) const {
            cudaEventRecord(stop_, stream.get());
            cudaEventSynchronize(stop_);

            float milliseconds = 0.0F;
            cudaEventElapsedTime(&milliseconds, start_, stop_);
            return static_cast<double>(milliseconds) / 1'000.0;
        }

    private:
        cudaEvent_t start_{};
        cudaEvent_t stop_{};
    };
} // namespace test
