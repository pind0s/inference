find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(inference_cpu STATIC
        backend.cpp
        bf16.cpp
        bf16.hpp
)

target_include_directories(inference_cpu
        PUBLIC
        ../..
)

target_compile_options(inference_cpu
        PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-march=x86-64-v4>
        $<$<COMPILE_LANGUAGE:CXX>:-mavx512bf16>
)

target_link_libraries(inference_cpu
        PUBLIC
        OpenMP::OpenMP_CXX
)
