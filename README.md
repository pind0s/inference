\[Work in progress\] \
LLM inference engine for Qwen3 models written in modern C++ with CUDA and AVX-512 CPU backends. Requires C++23 and CUDA
Toolkit 13.3

## Implemented
- Custom tensor class
- Fused kernels for Q/K RMSNorm RoPE and RMSNorm Add
- Flash attention optimized for decode for CUDA

## Limitations

- Currently only supports Qwen 3 models with BF16 weights. (tested only on Qwen3-0.6B BF16)
- No CUDA graphs and quantization
- No batching or optimized prefill
- CUDA backend uses cuBLAS for matmul

## Performance

Getting ~205 tokens/s with context size of 1024 on my RTX 3080. For reference, I'm getting ~300 tokens/s on vLLM. 
