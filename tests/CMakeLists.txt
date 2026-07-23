find_package(GTest CONFIG REQUIRED)

add_executable(inference_tests tests.cpp)

target_link_libraries(inference_tests
        PRIVATE
        inference_lib
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(inference_tests)