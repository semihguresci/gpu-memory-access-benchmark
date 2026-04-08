#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct OccupancyModelingExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string low_smem_shader_path;
    std::string medium_smem_shader_path;
    std::string high_smem_shader_path;
    bool verbose_progress = false;
};

struct OccupancyModelingExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

OccupancyModelingExperimentOutput
run_occupancy_modeling_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                  const OccupancyModelingExperimentConfig& config);
