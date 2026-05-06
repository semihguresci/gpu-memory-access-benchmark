#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct BvhNodeLayoutExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
};

struct BvhNodeLayoutExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

BvhNodeLayoutExperimentOutput run_bvh_node_layout_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                             const BvhNodeLayoutExperimentConfig& config);
