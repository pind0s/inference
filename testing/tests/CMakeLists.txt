find_package(Catch2 3 CONFIG REQUIRED)

add_executable(inference_tests add.cpp gemv.cpp qk_norm_rope.cpp rmsnorm.cpp rope.cpp sampling.cpp self_attention.cpp tokenizer.cpp)
target_include_directories(inference_tests PRIVATE ${PROJECT_SOURCE_DIR}/testing)
target_compile_definitions(inference_tests PRIVATE TEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources")
target_link_libraries(inference_tests PRIVATE inference_lib Catch2::Catch2WithMain CUDA::cudart)

include(Catch)
catch_discover_tests(inference_tests)
