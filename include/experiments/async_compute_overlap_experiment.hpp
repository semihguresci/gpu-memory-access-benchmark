#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct AsyncComputeOverlapExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
    bool verbose_progress = false;
};

struct AsyncComputeOverlapExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

AsyncComputeOverlapExperimentOutput
run_async_compute_overlap_experiment(VulkanContext& context, const BenchmarkRunner& runner,
                                     const AsyncComputeOverlapExperimentConfig& config);
