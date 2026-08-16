find_package(benchmark CONFIG REQUIRED)

add_executable(inference_benchmarks cpu/add.cpp cpu/rope.cpp cuda/matmul.cu cuda/rope.cu model/qwen3.cpp)

target_include_directories(inference_benchmarks PRIVATE ${PROJECT_SOURCE_DIR}/testing)

target_link_libraries(inference_benchmarks
        PRIVATE
        inference_lib
        inference_cpu
        benchmark::benchmark_main
)

if (MSVC)
    target_compile_options(inference_benchmarks PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif ()

set_target_properties(inference_benchmarks PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES native
)
