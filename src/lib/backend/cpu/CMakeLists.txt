find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(backend_cpu STATIC backend.cpp)

target_include_directories(backend_cpu PUBLIC ../..)

target_compile_options(backend_cpu PRIVATE -march=x86-64-v4 -mavx512bf16)


# clang-cl ignores -ffast-math (according to clanker)
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(backend_cpu PRIVATE /clang:-ffast-math)
else ()
    target_compile_options(backend_cpu PRIVATE -ffast-math)
endif ()

target_link_libraries(backend_cpu PRIVATE OpenMP::OpenMP_CXX)
