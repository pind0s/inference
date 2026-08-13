cmake_minimum_required(VERSION 4.2)
project(inference)
set(CMAKE_CXX_STANDARD 26)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS regex container_hash)
find_package(ICU REQUIRED COMPONENTS uc i18n)
find_package(argparse CONFIG REQUIRED)

add_subdirectory(src/cpu)

add_library(inference_lib STATIC src/tokenizer/tokenizer.cpp)

target_include_directories(inference_lib PUBLIC src)

target_link_libraries(inference_lib
        PUBLIC
        nlohmann_json::nlohmann_json
        Boost::regex
        ICU::uc
        ICU::i18n
        PRIVATE
        Boost::container_hash)


add_executable(inference src/main.cpp)

target_link_libraries(inference PRIVATE
        inference_lib
        inference_cpu
        argparse::argparse
)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
