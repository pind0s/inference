find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(inference_tests
        backend/add.cpp
        backend/matmul.cpp
        backend/rope.cpp
        backend/sampling.cpp
        backend/self_attention.cpp
        tokenizer.cpp
)

target_include_directories(inference_tests PRIVATE ${PROJECT_SOURCE_DIR}/testing ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_definitions(inference_tests PRIVATE TEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")
target_link_libraries(inference_tests PRIVATE inference_lib GTest::gmock_main)

gtest_discover_tests(inference_tests)
