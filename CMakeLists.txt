cmake_minimum_required(VERSION 4.2)
project(inference)

set(CMAKE_CXX_STANDARD 23)

add_executable(inference src/main.cpp
        src/util/args.hpp
        src/util/files.hpp
        src/tokenizer/tokenizer.cpp
        src/tokenizer/tokenizer.hpp)

find_package(argparse CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS regex container_hash)
find_package(ICU REQUIRED COMPONENTS uc i18n)

target_link_libraries(inference PRIVATE
        argparse::argparse
        nlohmann_json::nlohmann_json
        Boost::regex
        Boost::container_hash
        ICU::uc
        ICU::i18n)

target_include_directories(inference PRIVATE src)
# Tests
find_package(GTest CONFIG REQUIRED)

add_executable(tests tests/tests.cpp)

target_link_libraries(tests PRIVATE GTest::gtest GTest::gtest_main)
target_include_directories(tests PRIVATE src)

