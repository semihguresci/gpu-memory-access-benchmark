cmake_minimum_required(VERSION 3.21)

project(gpu_memory_layout_experiments VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_SHADERS "Compile GLSL shaders during the build" ON)
option(ENABLE_VULKAN_VALIDATION "Enable Vulkan validation by default at runtime" OFF)
include(CTest)

find_package(Vulkan REQUIRED)
include(FetchContent)

FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(cli11)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

include("${CMAKE_SOURCE_DIR}/cmake/experiments_manifest.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/generate_experiment_registry.cmake")

set(GENERATED_SOURCE_DIR "${CMAKE_BINARY_DIR}/generated")
set(GENERATED_EXPERIMENT_DIR "${GENERATED_SOURCE_DIR}/experiments")
file(MAKE_DIRECTORY "${GENERATED_EXPERIMENT_DIR}")

set(EXPERIMENT_REGISTRY_HEADER "${GENERATED_EXPERIMENT_DIR}/experiment_registry.hpp")
set(EXPERIMENT_REGISTRY_SOURCE "${GENERATED_EXPERIMENT_DIR}/experiment_registry.cpp")

generate_experiment_registry(
    MANIFEST_ENTRIES_VAR EXPERIMENT_MANIFEST_ENTRIES
    HEADER_TEMPLATE "${CMAKE_SOURCE_DIR}/cmake/templates/experiment_registry.hpp.in"
    SOURCE_TEMPLATE "${CMAKE_SOURCE_DIR}/cmake/templates/experiment_registry.cpp.in"
    OUTPUT_HEADER "${EXPERIMENT_REGISTRY_HEADER}"
    OUTPUT_SOURCE "${EXPERIMENT_REGISTRY_SOURCE}"
)

add_executable(gpu_memory_layout_experiments
    src/main.cpp
    "${EXPERIMENT_REGISTRY_SOURCE}"
    src/experiments/dispatch_basics_experiment.cpp
    src/experiments/local_size_sweep_experiment.cpp
    src/experiments/memory_copy_baseline_experiment.cpp
    src/experiments/sequential_indexing_experiment.cpp
    src/experiments/global_id_mapping_variants_experiment.cpp
    src/experiments/aos_soa_experiment.cpp
    src/experiments/aosoa_blocked_layout_experiment.cpp
    src/experiments/std430_std140_packed_experiment.cpp
    src/experiments/vec3_vec4_padding_costs_experiment.cpp
    src/experiments/scalar_type_width_sweep_experiment.cpp
    src/experiments/coalesced_vs_strided_experiment.cpp
    src/experiments/gather_access_pattern_experiment.cpp
    src/experiments/scatter_access_pattern_experiment.cpp
    src/experiments/read_reuse_cache_locality_experiment.cpp
    src/experiments/bandwidth_saturation_sweep_experiment.cpp
    src/experiments/shared_memory_tiling_experiment.cpp
    src/experiments/adapters/dispatch_basics_adapter.cpp
    src/experiments/adapters/local_size_sweep_adapter.cpp
    src/experiments/adapters/memory_copy_baseline_adapter.cpp
    src/experiments/adapters/sequential_indexing_adapter.cpp
    src/experiments/adapters/global_id_mapping_variants_adapter.cpp
    src/experiments/adapters/aos_soa_adapter.cpp
    src/experiments/adapters/aosoa_blocked_layout_adapter.cpp
    src/experiments/adapters/std430_std140_packed_adapter.cpp
    src/experiments/adapters/vec3_vec4_padding_costs_adapter.cpp
    src/experiments/adapters/scalar_type_width_sweep_adapter.cpp
    src/experiments/adapters/coalesced_vs_strided_adapter.cpp
    src/experiments/adapters/gather_access_pattern_adapter.cpp
    src/experiments/adapters/scatter_access_pattern_adapter.cpp
    src/experiments/adapters/read_reuse_cache_locality_adapter.cpp
    src/experiments/adapters/bandwidth_saturation_sweep_adapter.cpp
    src/experiments/adapters/shared_memory_tiling_adapter.cpp
    src/utils/app_options.cpp
    src/utils/json_exporter.cpp
    src/utils/vulkan_compute_utils.cpp
    src/utils/gpu_timestamp_timer.cpp
    src/vulkan_context.cpp
    src/utils/buffer_utils.cpp
    src/utils/scalar_type_width_utils.cpp
    src/benchmark_runner.cpp
)

target_include_directories(gpu_memory_layout_experiments
    PRIVATE
        include
        "${GENERATED_SOURCE_DIR}"
)

target_link_libraries(gpu_memory_layout_experiments
    PRIVATE
        Vulkan::Vulkan
        CLI11::CLI11
        nlohmann_json::nlohmann_json
)

target_compile_definitions(gpu_memory_layout_experiments
    PRIVATE
        $<$<BOOL:${ENABLE_VULKAN_VALIDATION}>:DEFAULT_ENABLE_VULKAN_VALIDATION=1>
)

if(MSVC)
    target_compile_options(gpu_memory_layout_experiments PRIVATE /W4 /permissive-)
    set_target_properties(gpu_memory_layout_experiments PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        VS_DEBUGGER_COMMAND_ARGUMENTS "--validation --experiment all --iterations 10 --warmup 3 --size 16M --output \"${CMAKE_SOURCE_DIR}/results/tables/all_run\""
    )
else()
    target_compile_options(gpu_memory_layout_experiments PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(SHADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/shaders")
set(SHADER_BINARY_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SHADER_BINARY_DIR}")

string(REPLACE "\\" "/" SHADER_SOURCE_DIR_NORMALIZED "${SHADER_SOURCE_DIR}")
string(REPLACE "\\" "/" SHADER_BINARY_DIR_NORMALIZED "${SHADER_BINARY_DIR}")

target_compile_definitions(gpu_memory_layout_experiments
    PRIVATE
        PROJECT_SHADER_SOURCE_DIR=\"${SHADER_SOURCE_DIR_NORMALIZED}\"
        PROJECT_SHADER_BINARY_DIR=\"${SHADER_BINARY_DIR_NORMALIZED}\"
)

if(BUILD_SHADERS)
    if(DEFINED Vulkan_GLSLC_EXECUTABLE AND EXISTS "${Vulkan_GLSLC_EXECUTABLE}")
        file(GLOB_RECURSE SHADER_SOURCES CONFIGURE_DEPENDS
            "${SHADER_SOURCE_DIR}/*.comp"
            "${SHADER_SOURCE_DIR}/*.vert"
            "${SHADER_SOURCE_DIR}/*.frag"
        )

        set(SPIRV_OUTPUTS)
        foreach(SHADER_FILE IN LISTS SHADER_SOURCES)
            get_filename_component(SHADER_NAME "${SHADER_FILE}" NAME)
            file(RELATIVE_PATH SHADER_RELATIVE_PATH "${SHADER_SOURCE_DIR}" "${SHADER_FILE}")
            set(SPIRV_FILE "${SHADER_BINARY_DIR}/${SHADER_NAME}.spv")

            add_custom_command(
                OUTPUT "${SPIRV_FILE}"
                COMMAND "${Vulkan_GLSLC_EXECUTABLE}" "${SHADER_FILE}" -o "${SPIRV_FILE}"
                DEPENDS "${SHADER_FILE}"
                COMMENT "Compiling shader ${SHADER_RELATIVE_PATH}"
                VERBATIM
            )

            list(APPEND SPIRV_OUTPUTS "${SPIRV_FILE}")
        endforeach()

        if(SPIRV_OUTPUTS)
            add_custom_target(compile_shaders ALL DEPENDS ${SPIRV_OUTPUTS})
            add_dependencies(gpu_memory_layout_experiments compile_shaders)
        endif()
    else()
        message(WARNING "glslc was not found by CMake. Set Vulkan_GLSLC_EXECUTABLE or disable BUILD_SHADERS.")
    endif()
endif()

include("${CMAKE_SOURCE_DIR}/cmake/tests.cmake")
configure_gpu_memory_layout_tests()
