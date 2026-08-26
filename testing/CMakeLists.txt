add_subdirectory(tests)
if (INFERENCE_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif ()
