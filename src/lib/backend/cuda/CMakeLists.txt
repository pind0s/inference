find_package(CUDAToolkit 13.3 REQUIRED)

add_library(backend_cuda STATIC backend.cu)

target_include_directories(backend_cuda PUBLIC ../..)

target_link_libraries(backend_cuda
        PUBLIC
        CUDA::cudart
        CUDA::cublas
)

if (MSVC)
    target_compile_options(backend_cuda
            PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif ()

set_target_properties(backend_cuda PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES native
)
