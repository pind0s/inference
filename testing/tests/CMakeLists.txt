find_package(GTest CONFIG REQUIRED)

add_executable(inference_tests
        cpu/add.cpp
        cpu/matmul.cpp
        cpu/rope.cpp
        cpu/sampling.cpp
        cpu/self_attention.cpp
        tokenizer.cpp
)

target_include_directories(inference_tests
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/testing
)

target_compile_definitions(inference_tests
        PRIVATE
        TEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources"
)

target_link_libraries(inference_tests
        PRIVATE
        inference_lib
        inference_cpu
        GTest::gmock_main
)

if (MSVC)
    target_compile_options(inference_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif ()

set_target_properties(inference_tests PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES native
)

include(GoogleTest)
gtest_discover_tests(inference_tests)
