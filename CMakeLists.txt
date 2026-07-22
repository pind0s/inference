cmake_minimum_required(VERSION 4.2)
project(template)

set(CMAKE_CXX_STANDARD 23)

add_executable(template src/main.cpp
        src/example.hpp)


# Tests
find_package(GTest CONFIG REQUIRED)

add_executable(tests tests/tests.cpp)

target_link_libraries(tests PRIVATE GTest::gtest GTest::gtest_main)
target_include_directories(tests PRIVATE src)

