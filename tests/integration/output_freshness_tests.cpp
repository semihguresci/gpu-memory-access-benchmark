#include "benchmark_runner.hpp"
#include "experiments/occupancy_modeling_experiment.hpp"
#include "utils/vulkan_compute_utils.hpp"
#include "vulkan_context.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

TEST(OutputFreshnessTests, RejectsNoopVariantAfterCorrectVariantPopulatesOutput) {
    VulkanContext context;
    if (!context.initialize(false) || !context.gpu_timestamps_supported()) {
        GTEST_SKIP() << "A Vulkan device with timestamps is required.";
    }
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context.physical_device(), &properties);
    std::string error;
    if (!VulkanComputeUtils::validate_compute_requirements(properties.limits, {256U, 1U, 1U}, 32768U, error)) {
        GTEST_SKIP() << error;
    }
    const std::filesystem::path noop_shader = std::filesystem::path(PROJECT_SHADER_BINARY_DIR) / "01_noop.comp.spv";
    ASSERT_TRUE(std::filesystem::exists(noop_shader));
    const BenchmarkRunner runner({.warmup_iterations = 1, .timed_iterations = 2});
    const auto output = run_occupancy_modeling_experiment(
        context, runner, {.max_buffer_bytes = 65536U, .medium_smem_shader_path = noop_shader.string()});
    ASSERT_FALSE(output.rows.empty());
    EXPECT_FALSE(output.all_points_correct);
    bool saw_noop = false;
    bool saw_correct = false;
    for (const auto& row : output.rows) {
        if (row.variant == "medium_smem") {
            saw_noop = true;
            EXPECT_FALSE(row.correctness_pass);
        } else {
            saw_correct = true;
            EXPECT_TRUE(row.correctness_pass);
        }
    }
    EXPECT_TRUE(saw_noop);
    EXPECT_TRUE(saw_correct);
}
