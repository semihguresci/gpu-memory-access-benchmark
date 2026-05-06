#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct RayFriendlyMemoryLayoutsExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
};

struct RayFriendlyMemoryLayoutsExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

RayFriendlyMemoryLayoutsExperimentOutput
run_ray_friendly_memory_layouts_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                           const RayFriendlyMemoryLayoutsExperimentConfig& config);
