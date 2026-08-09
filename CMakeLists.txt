cmake_minimum_required(VERSION 4.2)
project(inference)
set(CMAKE_CXX_STANDARD 26)


if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-march=x86-64-v4 -mavx512bf16)
else()
    message(FATAL_ERROR "project requires a Clang/GCC compiler")
endif()

find_package(nlohmann_json CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS regex container_hash)
find_package(ICU REQUIRED COMPONENTS uc i18n)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(argparse CONFIG REQUIRED)

add_library(inference_lib STATIC src/tokenizer/tokenizer.cpp)

target_include_directories(inference_lib PUBLIC src)

target_link_libraries(inference_lib
        PUBLIC
        nlohmann_json::nlohmann_json
        Boost::regex
        ICU::uc
        ICU::i18n
        OpenMP::OpenMP_CXX
        PRIVATE
        Boost::container_hash)


add_executable(inference src/main.cpp)

target_link_libraries(inference PRIVATE
        inference_lib
        argparse::argparse
)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
