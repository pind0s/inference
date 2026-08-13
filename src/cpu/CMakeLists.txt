if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR "The CPU backend requires a Clang/GCC compiler")
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(inference_cpu INTERFACE)

target_include_directories(inference_cpu
        INTERFACE
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_options(inference_cpu
        INTERFACE
        -march=x86-64-v4
        -mavx512bf16
)

target_link_libraries(inference_cpu
        INTERFACE
        OpenMP::OpenMP_CXX
)
