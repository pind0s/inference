cmake_minimum_required(VERSION 3.28)
project(inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)

find_package(CUDAToolkit REQUIRED)
find_package(argparse CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS regex)
find_package(ICU REQUIRED COMPONENTS uc i18n)

add_subdirectory(src/lib/backend/cpu)
add_subdirectory(src/lib/backend/cuda)

add_library(inference_lib STATIC
        src/lib/inference/generator.cpp
        src/lib/tensor/tensor.cpp
        src/lib/tokenizer/tokenizer.cpp)

target_include_directories(inference_lib
        PUBLIC
        src/lib
)

target_link_libraries(inference_lib
        PUBLIC
        nlohmann_json::nlohmann_json
        Boost::regex
        ICU::uc
        ICU::i18n
        PRIVATE
        backend_cpu
        backend_cuda
)

add_executable(inference src/entry/main.cpp)

target_link_libraries(inference PRIVATE inference_lib argparse::argparse)

include(CTest)
if (BUILD_TESTING)
    add_subdirectory(testing/tests)
endif ()
