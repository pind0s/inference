find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

function(configure_inference_test target)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/testing)
    target_link_libraries(${target} PRIVATE inference_lib GTest::gmock_main)
    gtest_discover_tests(${target})
endfunction()

add_executable(inference_backend_tests
        backend/add.cpp
        backend/matmul.cpp
        backend/rope.cpp
        backend/sampling.cpp
        backend/self_attention.cpp
)
configure_inference_test(inference_backend_tests)

add_executable(inference_tokenizer_tests tokenizer.cpp)
target_compile_definitions(inference_tokenizer_tests PRIVATE TEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")
configure_inference_test(inference_tokenizer_tests)
