\[Work in progress\] \
LLM inference engine written in modern C++ with custom kernels for AVX512 and CUDA. Requires C++23 and CUDA Toolkit 13.3

## Implemented

- Custom tensor class
- Fused kernels for Q/K RMSNorm RoPE and RmSNorm Add
- Flash attention for CUDA
- AVX-512 with BF16 specific instructions used in CPU kernels

## Limitations

- Currently only supports Qwen 3 models with BF16 weights. (tested only on Qwen 3 0.6B BF16)
- No CUDA graph and quantization
- No batching and prefill optimizations

