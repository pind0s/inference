cmake_minimum_required(VERSION 4.2)
project(inference)
set(CMAKE_CXX_STANDARD 23)

find_package(argparse CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS regex container_hash)
find_package(ICU REQUIRED COMPONENTS uc i18n)

add_library(inference_lib src/tokenizer/tokenizer.cpp)
target_include_directories(inference_lib PUBLIC src)

target_link_libraries(inference_lib PUBLIC
        nlohmann_json::nlohmann_json
        Boost::regex
        Boost::container_hash
        ICU::uc
        ICU::i18n)

add_executable(inference src/main.cpp)

target_link_libraries(inference PRIVATE
        inference_lib
        argparse::argparse
)


enable_testing()
add_subdirectory(tests)
