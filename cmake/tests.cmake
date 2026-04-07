include_guard(GLOBAL)

option(ENABLE_GPU_INTEGRATION_TESTS "Build GPU-dependent integration tests" OFF)

function(gpu_memory_layout_apply_test_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

function(gpu_memory_layout_add_gtest_target target_name test_labels)
    set(options)
    set(one_value_args)
    set(multi_value_args TEST_SOURCES APP_SOURCES EXTRA_LIBS)
    cmake_parse_arguments(PARSE_ARGV 2 ARG "${options}" "${one_value_args}" "${multi_value_args}")

    add_executable(${target_name}
        ${ARG_TEST_SOURCES}
        ${ARG_APP_SOURCES}
    )

    target_include_directories(${target_name}
        PRIVATE
            include
            tests
    )

    target_link_libraries(${target_name}
        PRIVATE
            GTest::gtest_main
            Vulkan::Vulkan
            CLI11::CLI11
            nlohmann_json::nlohmann_json
            ${ARG_EXTRA_LIBS}
    )

    gpu_memory_layout_apply_test_warnings(${target_name})

    include(GoogleTest)
    gtest_discover_tests(${target_name}
        DISCOVERY_MODE POST_BUILD
        PROPERTIES LABELS ${test_labels}
    )
endfunction()

function(configure_gpu_memory_layout_tests)
    if(NOT BUILD_TESTING)
        return()
    endif()

    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    set(unit_test_sources
        tests/unit/app_options_tests.cpp
        tests/unit/benchmark_runner_tests.cpp
        tests/unit/experiment_metrics_tests.cpp
        tests/unit/json_exporter_tests.cpp
        tests/unit/scratch_buffer_budget_tests.cpp
        tests/unit/scalar_type_width_utils_tests.cpp
        tests/unit/vulkan_compute_utils_tests.cpp
    )

    set(unit_app_sources
        src/benchmark_runner.cpp
        src/utils/app_options.cpp
        src/utils/json_exporter.cpp
        src/utils/scalar_type_width_utils.cpp
        src/utils/vulkan_compute_utils.cpp
    )

    gpu_memory_layout_add_gtest_target(
        gpu_memory_layout_unit_tests
        "unit"
        TEST_SOURCES ${unit_test_sources}
        APP_SOURCES ${unit_app_sources}
    )

    add_custom_target(gpu_memory_layout_tests DEPENDS gpu_memory_layout_unit_tests)

    if(ENABLE_GPU_INTEGRATION_TESTS)
        set(integration_test_sources
            tests/integration/experiment_smoke_tests.cpp
            tests/integration/vulkan_context_integration_tests.cpp
        )
        set(integration_app_sources
            src/vulkan_context.cpp
            src/utils/gpu_timestamp_timer.cpp
        )

        gpu_memory_layout_add_gtest_target(
            gpu_memory_layout_integration_tests
            "integration"
            TEST_SOURCES ${integration_test_sources}
            APP_SOURCES ${integration_app_sources}
        )

        target_compile_definitions(gpu_memory_layout_integration_tests
            PRIVATE
                GPU_MEMORY_LAYOUT_EXPERIMENTS_BINARY_DIR="$<TARGET_FILE_DIR:gpu_memory_layout_experiments>"
                GPU_MEMORY_LAYOUT_EXPERIMENTS_BINARY_NAME="$<TARGET_FILE_NAME:gpu_memory_layout_experiments>"
        )

        add_dependencies(gpu_memory_layout_integration_tests gpu_memory_layout_experiments)
        add_dependencies(gpu_memory_layout_tests gpu_memory_layout_integration_tests)
        add_dependencies(gpu_memory_layout_tests gpu_memory_layout_experiments)
    else()
        message(STATUS "GPU integration tests are disabled (set ENABLE_GPU_INTEGRATION_TESTS=ON to enable).")
    endif()
endfunction()
