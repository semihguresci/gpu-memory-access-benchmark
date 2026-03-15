#pragma once

#include "benchmark_runner.hpp"

#include <cstddef>
#include <string>
#include <vector>

class VulkanContext;

struct SequentialIndexingExperimentConfig {
    std::size_t max_buffer_bytes = 0;
    std::string shader_path;
};

struct SequentialIndexingExperimentOutput {
    std::vector<BenchmarkResult> summary_results;
    std::vector<BenchmarkMeasurementRow> rows;
    bool all_points_correct = true;
};

SequentialIndexingExperimentOutput run_sequential_indexing_experiment(VulkanContext& context,
                                                                      const BenchmarkRunner& runner,
                                                                      const SequentialIndexingExperimentConfig& config);
