find_package(CUDAToolkit 13.3 REQUIRED)

add_library(inference_cuda STATIC
        backend.cu
        kernels/matmul.cuh
)

target_include_directories(inference_cuda
        PUBLIC
        ../..
)

target_link_libraries(inference_cuda
        PUBLIC
        CUDA::cudart
        CUDA::cublas
)

if (MSVC)
    target_compile_options(inference_cuda
            PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif ()

set_target_properties(inference_cuda PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES native
)
