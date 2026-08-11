find_package(GTest CONFIG REQUIRED)

add_executable(inference_tests
        kernels/cpu/add.cpp
        kernels/cpu/matmul.cpp
        tokenizer.cpp)

target_include_directories(inference_tests PRIVATE .)

target_compile_definitions(inference_tests
        PRIVATE
        TEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources"
)

target_link_libraries(inference_tests
        PRIVATE
        inference_lib
        GTest::gmock_main
)

include(GoogleTest)
gtest_discover_tests(inference_tests)
