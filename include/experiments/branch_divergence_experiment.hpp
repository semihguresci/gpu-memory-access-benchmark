#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct BranchDivergenceExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::size_t scratch_size_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
};

struct BranchDivergenceExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

BranchDivergenceExperimentOutput run_branch_divergence_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                                                  const BranchDivergenceExperimentConfig& config);
