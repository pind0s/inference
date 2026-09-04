find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(backend_cpu STATIC backend.cpp)

target_include_directories(backend_cpu PUBLIC ../..)

target_compile_options(backend_cpu
        PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-march=x86-64-v4>
        $<$<COMPILE_LANGUAGE:CXX>:-mavx512bf16>
)

target_link_libraries(backend_cpu PRIVATE OpenMP::OpenMP_CXX)
