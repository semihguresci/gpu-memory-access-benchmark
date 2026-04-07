#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct ParallelReductionExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string global_atomic_shader_path;
    std::string shared_tree_shader_path;
    bool verbose_progress = false;
};

struct ParallelReductionExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

ParallelReductionExperimentOutput run_parallel_reduction_experiment(VulkanContext& context,
                                                                    const BenchmarkRunner& runner,
                                                                    const ParallelReductionExperimentConfig& config);
